import infrastructure
import enum


ODATA_API_BASE_URL = r"https://datahub.creodias.eu/odata/v1/Products"


class ODataAPIKeywords(enum.Enum):
    # ODataOptions
    FILTER = "$filter"
    ORDERBY = "$orderby"
    TOP = "$top"
    SKIP = "$skip"
    COUNT = "$count"
    EXPAND = "$expand"
    # Filters
    NAMECONTAINS = "contains(Name,'{}')"
    STARTDATE = "ContentDate/Start ge {}"
    ENDDATE = "ContentDate/Start le {}"
    INTERSECTS = "OData.CSC.Intersects(area=geography'SRID=4326;{}')"
    CLOUDCOVER = "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {})"
    # Other
    EXPANDATTRIBUTES = "Attributes"

    def __str__(self):
        return self.value


class ODataAPI(infrastructure.DataFinder):
    __resolver = {
        infrastructure.Satellites.SENTINEL1: "S1",
        infrastructure.Satellites.SENTINEL2: "S2",
        infrastructure.Satellites.LANDSAT5: "L5",
        infrastructure.Satellites.LANDSAT7: "L7",
        infrastructure.Satellites.LANDSAT8: "L8",
        infrastructure.Satellites.LANDSAT9: "L9",
        
        infrastructure.Sentinel1Products.GRD: "GRD",
        infrastructure.Sentinel1Products.SLC: "SLC",

        infrastructure.Sentinel2Products.L1C: "L1C",
        infrastructure.Sentinel2Products.L2A: "L2A",

        infrastructure.LandsatProducts.L1G: "L1G",
        infrastructure.LandsatProducts.L1T: "L1T",
        infrastructure.LandsatProducts.L1GT: "L1GT",

        infrastructure.LandsatTiers.T1: "T1",
        infrastructure.LandsatTiers.T2: "T2",

        infrastructure.EnvisatInstruments.MERIS: "MERIS",
        infrastructure.EnvisatInstruments.ASAR: "ASAR",
    }

    def __init__(self):
        super(ODataAPI, self).__init__(ODATA_API_BASE_URL)

    def search(self, *args, start_date=None, end_date=None, geometry=None, max_cloud_cover=None, orderby=None, top=1000, skip=None, count=None, expand=False):
        statements = { ODataAPIKeywords.FILTER: [] }
        for arg in args:
            resolved_arg = self.__resolve(arg)
            statement = ODataAPIKeywords.NAMECONTAINS.value.format(resolved_arg)
            statements[ODataAPIKeywords.FILTER].append(statement)
        if start_date:
            statement = ODataAPIKeywords.STARTDATE.value.format(start_date)
            statements[ODataAPIKeywords.FILTER].append(statement)
        if end_date:
            statement = ODataAPIKeywords.ENDDATE.value.format(end_date)
            statements[ODataAPIKeywords.FILTER].append(statement)
        if geometry:
            statement = ODataAPIKeywords.INTERSECTS.value.format(geometry.wkt)
            statements[ODataAPIKeywords.FILTER].append(statement)
        if max_cloud_cover:
            statement = ODataAPIKeywords.CLOUDCOVER.value.format(max_cloud_cover)
            statements[ODataAPIKeywords.FILTER].append(statement)
        statements[ODataAPIKeywords.FILTER] = " and ".join(statements[ODataAPIKeywords.FILTER])
        if orderby:
            statements[ODataAPIKeywords.ORDERBY] = self.__resolve(orderby)
        if top:
            statements[ODataAPIKeywords.TOP] = top
        if skip:
            statements[ODataAPIKeywords.SKIP] = skip
        if count:
            statements[ODataAPIKeywords.COUNT] = count
        if expand:
            statements[ODataAPIKeywords.EXPAND] = ODataAPIKeywords.EXPANDATTRIBUTES
        query_params = {str(option): statement for option, statement in statements.items()}
        data = self.search_raw(**query_params)
        return data

    def __resolve(self, arg):
        if isinstance(arg, str):
            return arg
        if arg not in self.__resolver:
            raise ValueError
        return self.__resolver[arg]
