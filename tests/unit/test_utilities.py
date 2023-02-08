from unittest import TestCase

from harmony.message import Message

from harmony_regridding_service.exceptions import InvalidTargetCRS
from harmony_regridding_service.utilities import (_has_all_attributes,
                                                  _has_consistent_dimension,
                                                  _is_geographic_crs,
                                                  get_file_mime_type,
                                                  has_self_consistent_grid,
                                                  has_dimensions,
                                                  has_scale_extents,
                                                  has_scale_sizes,
                                                  has_valid_crs,
                                                  has_valid_interpolation,
                                                  rgetattr)


class TestUtilities(TestCase):
    """ A class testing the harmony_regridding_service.utilities module. """

    def test_get_file_mime_type(self):
        """ Ensure a MIME type can be retrieved from an input file path. """
        with self.subTest('File with MIME type known by Python.'):
            self.assertEqual(get_file_mime_type('file.nc'),
                             'application/x-netcdf')

        with self.subTest('File with MIME type retrieved from dictionary.'):
            self.assertEqual(get_file_mime_type('file.nc4'),
                             'application/x-netcdf4')

        with self.subTest('File with entirely unknown MIME type.'):
            self.assertIsNone(get_file_mime_type('file.xyzzyx'))

    def test_has_valid_crs(self):
        """ Ensure the function correctly determines if the input Harmony
            message has a target Coordinate Reference System (CRS) that is
            compatible with the service. Currently this is either to not
            define the target CRS (assuming it to be geographic), or explicitly
            requesting geographic CRS via EPSG code or proj4 string.

        """
        with self.subTest('format = None returns True'):
            test_message = Message({})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = None returns True'):
            test_message = Message({'format': {}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "EPSG:4326" returns True'):
            test_message = Message({'format': {'crs': 'EPSG:4326'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "+proj=longlat" returns True'):
            test_message = Message({'format': {'crs': '+proj=longlat'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "4326" returns True'):
            test_message = Message({'format': {'crs': '4326'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('Non-geographic EPSG code returns False'):
            test_message = Message({'format': {'crs': 'EPSG:6933'}})
            self.assertFalse(has_valid_crs(test_message))

        with self.subTest('Non-geographic proj4 string returns False'):
            test_message = Message({'format': {'crs': '+proj=cea'}})
            self.assertFalse(has_valid_crs(test_message))

        with self.subTest('String that cannot be parsed raises exception'):
            test_message = Message({'format': {'crs': 'invalid CRS'}})

            with self.assertRaises(InvalidTargetCRS) as context:
                has_valid_crs(test_message)

            self.assertEqual(context.exception.message,
                             'Target CRS not supported: "invalid CRS"')

    def test_is_geographic_crs(self):
        """ Ensure function correctly determines if a supplied string resolves
            to a `pyproj.CRS` object with a geographic Coordinate Reference
            System (CRS). Exceptions arising from invalid CRS strings should
            also be handled.

        """
        with self.subTest('"EPSG:4326" returns True'):
            self.assertTrue(_is_geographic_crs('EPSG:4326'))

        with self.subTest('"+proj=longlat" returns True'):
            self.assertTrue(_is_geographic_crs('+proj=longlat'))

        with self.subTest('"4326" returns True'):
            self.assertTrue(_is_geographic_crs('4326'))

        with self.subTest('Non-geographic EPSG code returns False'):
            self.assertFalse(_is_geographic_crs('EPSG:6933'))

        with self.subTest('Non-geographic proj4 string returns False'):
            self.assertFalse(_is_geographic_crs('+proj=cea'))

        with self.subTest('String that cannot be parsed raises exception'):
            with self.assertRaises(InvalidTargetCRS) as context:
                _is_geographic_crs('invalid CRS')

            self.assertEqual(context.exception.message,
                             'Target CRS not supported: "invalid CRS"')

    def test_has_valid_interpolation(self):
        """ Ensure that the function correctly determines if the supplied
            Harmony message either omits the `format.interpolation` attribute,
            or specifies EWA via a fully spelled-out string. The TRT-210 MVP
            only allows for interpolation using EWA.

        """
        with self.subTest('format = None returns True'):
            test_message = Message({})
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('format.interpolation = None returns True'):
            test_message = Message({'format': {}})
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('EWA (spelled fully) returns True'):
            test_message = Message({
                'format': {'interpolation': 'Elliptical Weighted Averaging'}
            })
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('Unexpected interpolation returns False'):
            test_message = Message({'format': {'interpolation': 'Bilinear'}})
            self.assertFalse(has_valid_interpolation(test_message))

    def test_self_has_self_consistent_grid(self):
        """ Ensure that the function correctly determines if the supplied
            Harmony message defines a valid target grid. This should either:

            * Specify scaleExtent and 1 of:
              * Height and Width
              * Scale sizes (in the x and y horizontal spatial dimensions)
            * Specify all three of the above, but the values must be consistent
              with one another.

        """
        valid_scale_extents = {'x': {'min': -180, 'max': 180},
                               'y': {'min': -90, 'max': 90}}

        valid_scale_sizes = {'x': 0.5, 'y': 1.0}
        valid_height = 181
        valid_width = 721

        with self.subTest('format = None returns False'):
            test_message = Message({})
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('All grid parameters = None returns False'):
            test_message = Message({'format': {}})
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('Only scaleExtent returns False'):
            test_message = Message({
                'format': {'scaleExtents': valid_scale_extents}
            })
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('Only scaleSize returns False'):
            test_message = Message({
                'format': {'scaleSize': valid_scale_sizes}
            })
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('Only dimensions (height and width) returns False'):
            test_message = Message({
                'format': {'height': valid_height, 'width': valid_width}
            })
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('Dimensions and scaleSize returns False'):
            # Need the extents to know where to put the pixels!
            test_message = Message({
                'format': {'height': valid_height,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width}
            })
            self.assertFalse(has_self_consistent_grid(test_message))

        with self.subTest('Dimensions and scaleExtent returns True'):
            test_message = Message({
                'format': {'height': valid_height,
                           'scaleExtent': valid_scale_extents,
                           'width': valid_width}
            })
            self.assertTrue(has_self_consistent_grid(test_message))

        with self.subTest('scaleExtent and scaleSize returns True'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            self.assertTrue(has_self_consistent_grid(test_message))

        with self.subTest('All three, and consistent returns True'):
            test_message = Message({
                'format': {'height': valid_height,
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width}
            })
            self.assertTrue(has_self_consistent_grid(test_message))

        with self.subTest('All three, but inconsistent returns False'):
            test_message = Message({
                'format': {'height': valid_height + 150,
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width - 150}
            })
            self.assertFalse(has_self_consistent_grid(test_message))

    def test_has_consistent_dimension(self):
        """ Ensure, given a scale size (resolution), scale extent (range) and
            dimension size, the function can correctly determine if all three
            values are consistent with one another.

        """
        valid_scale_extents = {'x': {'min': -180, 'max': 180},
                               'y': {'min': -90, 'max': 90}}

        valid_scale_sizes = {'x': 0.5, 'y': 1.0}
        valid_height = 181
        valid_width = 721

        with self.subTest('Consistent x dimension returns True'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width}
            })
            self.assertTrue(_has_consistent_dimension(test_message, 'x'))

        with self.subTest('Consistent y dimension returns True'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'height': valid_height}
            })
            self.assertTrue(_has_consistent_dimension(test_message, 'y'))

        with self.subTest('Inconsistent x dimension returns False'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width + 100}
            })
            self.assertFalse(_has_consistent_dimension(test_message, 'x'))

        with self.subTest('Inconsistent y dimension returns False'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'height': valid_height + 100}
            })
            self.assertFalse(_has_consistent_dimension(test_message, 'y'))

    def test_has_all_attributes(self):
        """ Ensure that the function returns the correct boolean value
            indicating if all requested attributes are present in the supplied
            object, and have non-None values.

        """
        test_message = Message({'format': {'scaleSize': {'x': 0.5, 'y': 0.5}}})

        with self.subTest('All attributes present returns True'):
            self.assertTrue(_has_all_attributes(test_message,
                                                ['format.scaleSize.x',
                                                 'format.scaleSize.y']))

        with self.subTest('Some attributes present returns False'):
            self.assertFalse(_has_all_attributes(test_message,
                                                 ['format.scaleSize.x',
                                                  'format.height']))

        with self.subTest('No attributes present returns False'):
            self.assertFalse(_has_all_attributes(test_message,
                                                 ['format.height',
                                                  'format.width']))

    def test_has_scale_sizes(self):
        """ Ensure the function correctly identifies whether the supplied
            Harmony message contains both an x and y scale size.

        """
        with self.subTest('Scale sizes present returns True'):
            test_message = Message({
                'format': {'scaleSize': {'x': 0.5, 'y': 0.5}}
            })
            self.assertTrue(has_scale_sizes(test_message))

        with self.subTest('scaleSize.x = None returns False'):
            test_message = Message({'format': {'scaleSize': {'y': 0.5}}})
            self.assertFalse(has_scale_sizes(test_message))

        with self.subTest('scaleSize.y = None returns False'):
            test_message = Message({'format': {'scaleSize': {'x': 0.5}}})
            self.assertFalse(has_scale_sizes(test_message))

        with self.subTest('Both scaleSizes = None returns False'):
            test_message = Message({'format': {'scaleSize': {}}})
            self.assertFalse(has_scale_sizes(test_message))

        with self.subTest('format = None returns False'):
            test_message = Message({})
            self.assertFalse(has_scale_sizes(test_message))

    def test_has_scale_extents(self):
        """ Ensure the function correctly identifies whether the supplied
            Harmony message contains all required elements in the
            `format.scaleExtent` attribute. This includes minima and maxima for
            both the x and y horizontal spatial dimensions of the target grid.

        """
        x_extents = {'min': -180, 'max': 180}
        y_extents = {'min': -90, 'max': 90}

        with self.subTest('Scale extents present returns True'):
            test_message = Message({
                'format': {'scaleExtent': {'x': x_extents, 'y': y_extents}}
            })
            self.assertTrue(has_scale_extents(test_message))

        with self.subTest('scaleExtent.x.min = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': {'max': 180}, 'y': y_extents}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.x.max = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': {'min': -180}, 'y': y_extents}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.x min and max = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': {}, 'y': y_extents}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.y.min = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': x_extents, 'y': {'max': 90}}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.y.max = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': x_extents, 'y': {'min': -90}}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.y min and max = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': x_extents, 'y': {}}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('All scaleExtent values = None returns False'):
            test_message = Message({
                'format': {'scaleExtent': {'x': {}, 'y': {}}}
            })
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('scaleExtent.x and scaleExtent.y = None returns False'):
            test_message = Message({'format': {'scaleExtent': {}}})
            self.assertFalse(has_scale_extents(test_message))

        with self.subTest('format = None returns False'):
            test_message = Message({})
            self.assertFalse(has_scale_extents(test_message))

    def test_has_dimensions(self):
        """ Ensure the function correctly validates whether the supplied
            Harmony message contains both a height an width for the target
            grid.

        """
        with self.subTest('height and width present returns True'):
            test_message = Message({'format': {'height': 100, 'width': 50}})
            self.assertTrue(has_dimensions(test_message))

        with self.subTest('height = None returns False'):
            test_message = Message({'format': {'width': 50}})
            self.assertFalse(has_dimensions(test_message))

        with self.subTest('width = None returns False'):
            test_message = Message({'format': {'height': 100}})
            self.assertFalse(has_dimensions(test_message))

        with self.subTest('height = None and width = None returns False'):
            test_message = Message({'format': {}})
            self.assertFalse(has_dimensions(test_message))

        with self.subTest('format = None returns False'):
            test_message = Message({})
            self.assertFalse(has_dimensions(test_message))

    def test_rgetattr(self):
        """ Ensure that the recursive function can retrieve nested attributes
            and uses the default argument when required.

        """
        class Child:
            def __init__(self, name):
                self.name = name

        class Parent:
            def __init__(self, name, child_name):
                self.name = name
                self.child = Child(child_name)

        test_parent = Parent('parent_name', 'child_name')

        with self.subTest('Parent level attribute'):
            self.assertEqual(rgetattr(test_parent, 'name'), 'parent_name')

        with self.subTest('Nested attribute'):
            self.assertEqual(rgetattr(test_parent, 'child.name'), 'child_name')

        with self.subTest('Missing parent with default'):
            self.assertEqual(rgetattr(test_parent, 'absent', 'default'),
                             'default')

        with self.subTest('Missing child attribute with default'):
            self.assertEqual(rgetattr(test_parent, 'child.absent', 'default'),
                             'default')

        with self.subTest('Child requested, parent missing, default'):
            self.assertEqual(
                rgetattr(test_parent, 'none.something', 'default'),
                'default'
            )

        with self.subTest('Missing parent, with no default'):
            self.assertIsNone(rgetattr(test_parent, 'absent'))

        with self.subTest('Missing child, with no default'):
            self.assertIsNone(rgetattr(test_parent, 'child.absent'))

        with self.subTest('Child requested, parent missing, no default'):
            self.assertIsNone(rgetattr(test_parent, 'absent.something'))
