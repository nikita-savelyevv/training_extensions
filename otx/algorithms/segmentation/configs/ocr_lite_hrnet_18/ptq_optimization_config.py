"""PTQ config file."""
from nncf import IgnoredScope
from nncf.common.quantization.structs import QuantizationPreset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.range_estimator import (
    AggregatorType,
    RangeEstimatorParameters,
    StatisticsCollectorParameters,
    StatisticsType,
)

advanced_parameters = AdvancedQuantizationParameters(
    activations_range_estimator_params=RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MIN, quantile_outlier_prob=1e-4
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MAX, quantile_outlier_prob=1e-4
        ),
    )
)

preset = QuantizationPreset.PERFORMANCE

ignored_scope = IgnoredScope(
    names=[
        "Mul_113",
        "Mul_133",
        "Mul_260",
        "Mul_280",
        "Add_374",
        "Mul_437",
        "Mul_457",
        "Mul_584",
        "Mul_604",
        "Add_698",
        "Mul_779",
        "Mul_799",
        "Mul_819",
        "Mul_997",
        "Mul_1017",
        "Add_1167",
        "Mul_1037",
        "Add_1188",
        "Add_1216",
        "Mul_1303",
        "Mul_1323",
        "Mul_1343",
        "Mul_1521",
        "Mul_1541",
        "Add_1691",
        "Mul_1561",
        "Add_1712",
        "Add_1740",
        "Mul_1827",
        "Mul_1847",
        "Mul_1867",
        "Mul_2045",
        "Mul_2065",
        "Add_2215",
        "Mul_2085",
        "Add_2236",
        "Add_2264",
        "Mul_2351",
        "Mul_2371",
        "Mul_2391",
        "Mul_2569",
        "Mul_2589",
        "Add_2739",
        "Mul_2609",
        "Add_2760",
        "Add_2788",
        "Mul_2893",
        "Mul_2913",
        "Mul_2933",
        "Mul_2953",
        "Mul_3182",
        "Mul_3202",
        "Add_3408",
        "Mul_3222",
        "Add_3429",
        "Mul_3242",
        "Add_3450",
        "Add_3478",
        "Add_3499",
        "Add_3536",
        "Mul_3650",
        "Mul_3670",
        "Mul_3690",
        "Mul_3710",
        "Mul_3939",
        "Mul_3959",
        "Add_4165",
        "Mul_3979",
        "Add_4186",
        "Mul_3999",
        "Add_4207",
        "Add_4235",
        "Add_4256",
        "Add_4293",
        "Add_4345",
        "Add_4368",
        "Add_4391",
    ]
)
