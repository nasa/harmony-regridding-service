""" Utility modules for use within the Harmony Regridding Service. These
    include MIME type determination and basic components of message validation.

"""
from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from typing import Any, List, Optional

from harmony.message import Message
from numpy import divide, isclose
from pyproj import CRS
from pyproj.exceptions import CRSError

from harmony_regridding_service.exceptions import InvalidTargetCRS


KNOWN_MIME_TYPES = {'.nc4': 'application/x-netcdf4',
                    '.h5': 'application/x-hdf5',
                    '.hdf5': 'application/x-hdf5'}
VALID_INTERPOLATION_METHODS = ('Elliptical Weighted Averaging', )


def get_file_mime_type(file_name: str) -> Optional[str]:
    """ This function tries to infer the MIME type of a file string. If the
        `mimetypes.guess_type` function cannot guess the MIME type of the
        granule, a dictionary of known file types is checked using the file
        extension. That dictionary only contains keys for MIME types that
        `mimetypes.guess_type` cannot resolve.

    """
    mime_type = guess_mime_type(file_name, False)

    if not mime_type or mime_type[0] is None:
        mime_type = (KNOWN_MIME_TYPES.get(splitext(file_name)[1].lower()),
                     None)

    return mime_type[0]


def has_valid_crs(message: Message) -> bool:
    """ Check the target Coordinate Reference System (CRS) in the Harmony
        message is compatible with the CRS types supported by the regridder. In
        the MVP, only a geographic CRS is supported, so `Message.format.crs`
        should either be undefined or specify a geographic CRS.

    """
    target_crs = rgetattr(message, 'format.crs')
    return target_crs is None or _is_geographic_crs(target_crs)


def _is_geographic_crs(crs_string: str) -> bool:
    """ Use pyproj to ascertain if the supplied Coordinate Reference System
        (CRS) is geographic.

    """
    try:
        crs = CRS(crs_string)
        is_geographic = crs.is_geographic
    except CRSError as exception:
        raise InvalidTargetCRS(crs_string) from exception

    return is_geographic


def has_valid_interpolation(message: Message) -> bool:
    """ Check the interpolation method in the input Harmony message is
        compatible with the methods supported by the regridder. In the MVP,
        only the EWA algorithm is used, so either the interpolation should be
        unspecified in the message, or it should be the string
        "Elliptical Weighted Averaging".

    """
    interpolation_method = rgetattr(message, 'format.interpolation')

    return (interpolation_method is None
            or interpolation_method in VALID_INTERPOLATION_METHODS)


def has_self_consistent_grid(message: Message) -> bool:
    """ Check the input Harmony message provides enough information to fully
        define the target grid. At minimum the message should contain the scale
        extents (minimum and maximum values) in the horizontal spatial
        dimensions and one of the following two pieces of information:

        * Message.format.scaleSize - defining the x and y pixel size.
        * Message.format.height and Message.format.width - the number of pixels
          in the x and y dimension.

        If all three pieces of information are supplied, they will be checked
        to ensure they are consistent with one another.

        If scaleExtent and scaleSize are defined, along with only one of
        height or width, the grid will be considered consistent if the three
        values for scaleExtent, scaleSize and specified dimension length,
        height or width, are consistent.

    """
    if (
        has_scale_extents(message) and has_scale_sizes(message)
        and has_dimensions(message)
    ):
        consistent_grid = (_has_consistent_dimension(message, 'x')
                           and _has_consistent_dimension(message, 'y'))
    elif (
        has_scale_extents(message) and has_scale_sizes(message)
        and rgetattr(message, 'format.height') is not None
    ):
        consistent_grid = _has_consistent_dimension(message, 'y')
    elif (
        has_scale_extents(message) and has_scale_sizes(message)
        and rgetattr(message, 'format.width') is not None
    ):
        consistent_grid = _has_consistent_dimension(message, 'x')
    elif (
        has_scale_extents(message)
        and (has_scale_sizes(message) or has_dimensions(message))
    ):
        consistent_grid = True
    else:
        consistent_grid = False

    return consistent_grid


def has_dimensions(message: Message) -> bool:
    """ Ensure the supplied Harmony message contains values for height and
        width of the target grid, which define the sizes of the x and y
        horizontal spatial dimensions.

    """
    return _has_all_attributes(message, ['format.height', 'format.width'])


def has_scale_extents(message: Message) -> bool:
    """ Ensure the supplied Harmony message contains values for the minimum and
        maximum extents of the target grid in both the x and y dimensions.

    """
    scale_extent_attributes = ['format.scaleExtent.x.min',
                               'format.scaleExtent.x.max',
                               'format.scaleExtent.y.min',
                               'format.scaleExtent.y.max']

    return _has_all_attributes(message, scale_extent_attributes)


def has_scale_sizes(message: Message) -> bool:
    """ Ensure the supplied Harmony message contains values for the x and y
        horizontal scale sizes for the target grid.

    """
    scale_size_attributes = ['format.scaleSize.x', 'format.scaleSize.y']
    return _has_all_attributes(message, scale_size_attributes)


def _has_all_attributes(message: Message, attributes: List[str]) -> bool:
    """ Ensure that the supplied Harmony message has non-None attribute values
        for all the listed attributes.

    """
    return all(rgetattr(message, attribute_name) is not None
               for attribute_name in attributes)


def _has_consistent_dimension(message: Message, dimension_name: str) -> bool:
    """ Ensure a grid dimension has consistent values for the scale extent
        (e.g., minimum and maximum values), scale size (resolution) and
        dimension length (e.g., width or height). For the grid x dimension, the
        calculation is as follows:

        scaleSize.x = (scaleExtent.x.max - scaleExtent.x.min) / (width)

        The message scale sizes is compared to that calculated as above, to
        ensure it is within a relative tolerance (1 x 10^-3).

    """
    message_scale_size = getattr(message.format.scaleSize, dimension_name)
    scale_extent = getattr(message.format.scaleExtent, dimension_name)

    if dimension_name == 'x':
        dimension_elements = message.format.width
    else:
        dimension_elements = message.format.height

    derived_scale_size = divide((scale_extent.max - scale_extent.min),
                                dimension_elements)

    return isclose(message_scale_size, derived_scale_size, rtol=1e-3, atol=0)


def rgetattr(input_object: Any, requested_attribute: str, *args) -> Any:
    """ This is a recursive version of the inbuilt `getattr` method, such that
        it can be called to retrieve nested attributes. For example:
        the Message.subset.shape within the input Harmony message.

        Note, if a default value is specified, this will be returned if any
        attribute in the specified chain is absent from the supplied object.
        Alternatively, if an absent attribute is specified and no default value
        if given in the function call, this function will return `None`.

    """
    if len(args) == 0:
        args = (None, )

    if '.' not in requested_attribute:
        result = getattr(input_object, requested_attribute, *args)
    else:
        attribute_pieces = requested_attribute.split('.')
        result = rgetattr(getattr(input_object, attribute_pieces[0], *args),
                          '.'.join(attribute_pieces[1:]), *args)

    return result
