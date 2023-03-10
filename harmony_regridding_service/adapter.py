""" `RegriddingServiceAdapter` for Harmony Regridding Service.

    The class in this file is the top level of abstraction for a service that
    will accept a gridded input file (e.g., L3/L4) and transform the data to
    another grid as specified in the input Harmony message.

"""
from os.path import basename
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional

from harmony import BaseHarmonyAdapter
from harmony.message import Source as HarmonySource
from harmony.util import (bbox_to_geometry, download, generate_output_filename,
                          stage)
from pystac import Asset, Catalog, Item


from harmony_regridding_service.exceptions import (InvalidInterpolationMethod,
                                                   InvalidTargetCRS,
                                                   InvalidTargetGrid)
from harmony_regridding_service.regridding_service import regrid
from harmony_regridding_service.utilities import (get_file_mime_type,
                                                  has_valid_crs,
                                                  has_valid_interpolation,
                                                  has_self_consistent_grid)


class RegriddingServiceAdapter(BaseHarmonyAdapter):
    """ This class extends the BaseHarmonyAdapter class from the
        harmony-service-lib package to implement regridding operations.

    """
    def __init__(self, message, catalog=None, config=None):
        super().__init__(message, catalog=catalog, config=config)
        self.cache = {'grids': {}}

    def invoke(self) -> Catalog:
        """ Adds validation to process_item based invocations. """
        self.validate_message()
        return super().invoke()

    def validate_message(self):
        """ Validates that the contents of the Harmony message provides all
            necessary parameters.

            For an input Harmony message to be considered valid it must:

            * Contain a valid target grid, with `format.scaleExtent` and either
              `format.scaleSize` or both `format.height` and `format.width`
              fully populated.
            * Not specify an incompatible target CRS. Initially, the target CRS
              is limited to geographic. The message should either specify a
              geographic CRS, or not specify one at all.
            * Not specify an incompatible interpolation method. Initially, the
              Harmony Regridding Service will use Elliptical Weighted Averaging
              to interpolate when needed. The message should either specify
              this interpolation method, or not specify one at all.

        """
        if not has_valid_crs(self.message):
            raise InvalidTargetCRS(self.message.format.crs)

        if not has_valid_interpolation(self.message):
            raise InvalidInterpolationMethod(self.message.format.interpolation)

        if not has_self_consistent_grid(self.message):
            raise InvalidTargetGrid()

    def process_item(self, item: Item, source: HarmonySource) -> Item:
        """ Processes a single input STAC item. """
        try:
            working_directory = mkdtemp()
            results = item.clone()
            results.assets = {}

            asset = next((item_asset for item_asset in item.assets.values()
                          if 'data' in (item_asset.roles or [])))

            # Download the input:
            input_filepath = download(asset.href, working_directory,
                                      logger=self.logger, cfg=self.config,
                                      access_token=self.message.accessToken)

            transformed_file_name = regrid(self, input_filepath, source)

            # Stage the transformed output:
            transformed_mime_type = get_file_mime_type(transformed_file_name)
            staged_url = self.stage_output(transformed_file_name, asset.href,
                                           transformed_mime_type)

            return self.create_output_stac_item(item, staged_url,
                                                transformed_mime_type)
        except Exception as exception:
            self.logger.exception(exception)
            raise exception
        finally:
            rmtree(working_directory)

    def stage_output(self, transformed_file: str, input_file: str,
                     transformed_mime_type: Optional[str]) -> str:
        """ Generate an output file name based on the input asset URL and the
            operations performed to produce the output. Use this name to stage
            the output in the S3 location specified in the input Harmony
            message.

        """
        output_file_name = generate_output_filename(input_file,
                                                    is_regridded=True)

        return stage(transformed_file, output_file_name, transformed_mime_type,
                     location=self.message.stagingLocation,
                     logger=self.logger, cfg=self.config)

    def create_output_stac_item(self, input_stac_item: Item, staged_url: str,
                                transformed_mime_type: str) -> Item:
        """ Create an output STAC item used to access the transformed and
            staged output in S3.

        """
        output_stac_item = input_stac_item.clone()
        output_stac_item.assets = {}
        # The output bounding box will vary by grid, so the following line
        # will need to be updated when the service has access to the output
        # grid specification.
        output_stac_item.bbox = input_stac_item.bbox
        output_stac_item.geometry = bbox_to_geometry(output_stac_item.bbox)

        output_stac_item.assets['data'] = Asset(
            staged_url, title=basename(staged_url),
            media_type=transformed_mime_type, roles=['data']
        )

        return output_stac_item
