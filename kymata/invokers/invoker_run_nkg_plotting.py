from logging import basicConfig, INFO
from pathlib import Path
from os import path

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.expression import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict


def main():
    function_family_type = "standard"  # 'standard' or 'ANN'
    path_to_nkg_files = Path(
        Path(__file__).parent.parent.parent, "kymata-core-data", "output"
    )

    # template invoker for printing out expression set .nkgs

    if function_family_type == "standard":
        expression_data = load_expression_set(
            Path(path_to_nkg_files, 'no_06_11_derange_10', 'left', "11_transforms_gridsearch.nkg")
        )
        expression_data.rename({'IL': 'IL_left',
                                'IL1': 'IL1_left',
                                'IL2': 'IL2_left',
                                'IL3': 'IL3_left',
                                'IL4': 'IL4_left',
                                'IL5': 'IL5_left',
                                'IL6': 'IL6_left',
                                'IL7': 'IL7_left',
                                'IL8': 'IL8_left',
                                'IL9': 'IL9_left',
                                'STL': 'STL_left'
                                })
        expression_data += load_expression_set(
            Path(path_to_nkg_files, 'no_06_11_derange_10', 'right', "11_transforms_gridsearch.nkg")
        )
        expression_data.rename({'IL': 'IL_right',
                                'IL1': 'IL1_right',
                                'IL2': 'IL2_right',
                                'IL3': 'IL3_right',
                                'IL4': 'IL4_right',
                                'IL5': 'IL5_right',
                                'IL6': 'IL6_right',
                                'IL7': 'IL7_right',
                                'IL8': 'IL8_right',
                                'IL9': 'IL9_right',
                                'STL': 'STL_right'
                                })
        expression_data += load_expression_set(
            Path(path_to_nkg_files, 'no_06_11_derange_10', 'added', "11_transforms_gridsearch.nkg")
        )

        fig = expression_plot(
            expression_data,
            color={
                "IL_left": "#b11e34",
                "IL1_left": "#b11e34",
                "IL2_left": "#b11e34",
                "IL3_left": "#b11e34",
                "IL4_left": "#b11e34",
                "IL5_left": "#b11e34",
                "IL6_left": "#b11e34",
                "IL7_left": "#b11e34",
                "IL8_left": "#b11e34",
                "IL9_left": "#b11e34",
                "STL_left": "#b11e34",
                "IL_right": "#1e6fb1",
                "IL1_right": "#1e6fb1",
                "IL2_right": "#1e6fb1",
                "IL3_right": "#1e6fb1",
                "IL4_right": "#1e6fb1",
                "IL5_right": "#1e6fb1",
                "IL6_right": "#1e6fb1",
                "IL7_right": "#1e6fb1",
                "IL8_right": "#1e6fb1",
                "IL9_right": "#1e6fb1",
                "STL_right": "#1e6fb1",
                "IL": "#b1a71e",
                "IL1": "#b1a71e",
                "IL2": "#b1a71e",
                "IL3": "#b1a71e",
                "IL4": "#b1a71e",
                "IL5": "#b1a71e",
                "IL6": "#b1a71e",
                "IL7": "#b1a71e",
                "IL8": "#b1a71e",
                "IL9": "#b1a71e",
                "STL": "#b1a71e",
            },
            use_sensor_layout='CBU',
            # minimap="standard",
            # ylim=-200,
        )

        fig.savefig(Path(path_to_nkg_files, "expression_plot_left_right_added.png"))

    elif function_family_type == "ANN":
        path_to_nkg_files = Path(path_to_nkg_files, "whisper/encoder_all_der_5")

        expression_data = load_expression_set(
            Path(path_to_nkg_files, "model.encoder.conv1_511_gridsearch.nkg")
        )
        expression_data += load_expression_set(
            Path(path_to_nkg_files, "model.encoder.conv2_511_gridsearch.nkg")
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.0.final_layer_norm_511_gridsearch.nkg",
            )
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.1.final_layer_norm_511_gridsearch.nkg",
            )
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.2.final_layer_norm_511_gridsearch.nkg",
            )
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.3.final_layer_norm_511_gridsearch.nkg",
            )
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.4.final_layer_norm_511_gridsearch.nkg",
            )
        )
        expression_data += load_expression_set(
            Path(
                path_to_nkg_files,
                "model.encoder.layers.5.final_layer_norm_511_gridsearch.nkg",
            )
        )

        conv1_list = []
        conv2_list = []
        encoder0_list = []
        encoder1_list = []
        encoder2_list = []
        encoder3_list = []
        encoder4_list = []
        encoder5_list = []

        # Loop through the range from 0 to 511
        for i in range(512):
            conv1 = f"model.encoder.conv1_{i}"
            conv1_list.append(conv1)

        for i in range(512):
            conv2 = f"model.encoder.conv2_{i}"
            conv2_list.append(conv2)

        for i in range(512):
            encoder0 = f"model.encoder.layers.0.final_layer_norm_{i}"
            encoder0_list.append(encoder0)

        for i in range(512):
            encoder1 = f"model.encoder.layers.1.final_layer_norm_{i}"
            encoder1_list.append(encoder1)

        for i in range(512):
            encoder2 = f"model.encoder.layers.2.final_layer_norm_{i}"
            encoder2_list.append(encoder2)

        for i in range(512):
            encoder3 = f"model.encoder.layers.3.final_layer_norm_{i}"
            encoder3_list.append(encoder3)

        for i in range(512):
            encoder4 = f"model.encoder.layers.4.final_layer_norm_{i}"
            encoder4_list.append(encoder4)

        for i in range(512):
            encoder5 = f"model.encoder.layers.5.final_layer_norm_{i}"
            encoder5_list.append(encoder5)

        fig = expression_plot(
            expression_data,
            ylim=-400,
            xlims=(-200, 800),
            save_to=Path(
                Path(path.abspath("")).parent,
                "kymata-core/kymata-core-data",
                "output/encoder_all.jpg",
            ),
            show_legend=True,
            color=constant_color_dict(conv1_list, color="red")
            | constant_color_dict(conv2_list, color="green")
            | constant_color_dict(encoder0_list, color="blue")
            | constant_color_dict(encoder1_list, color="cyan")
            | constant_color_dict(encoder2_list, color="magenta")
            | constant_color_dict(encoder3_list, color="yellow")
            | constant_color_dict(encoder4_list, color="orange")
            | constant_color_dict(encoder5_list, color="purple"),
            legend_display=legend_display_dict(conv1_list, "Conv layer 1")
            | legend_display_dict(conv1_list, "Conv layer 1")
            | legend_display_dict(conv2_list, "Conv layer 2")
            | legend_display_dict(encoder0_list, "Encoder layer 1")
            | legend_display_dict(encoder1_list, "Encoder layer 2")
            | legend_display_dict(encoder2_list, "Encoder layer 3")
            | legend_display_dict(encoder3_list, "Encoder layer 4")
            | legend_display_dict(encoder4_list, "Encoder layer 5")
            | legend_display_dict(encoder5_list, "Encoder layer 6"),
        )

        fig.savefig(Path(path_to_nkg_files, "expression_plot.png"))


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
