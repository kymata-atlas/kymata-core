from logging import basicConfig, INFO
from pathlib import Path
from os import path

from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set
from kymata.plot.expression import expression_plot, legend_display_dict
from kymata.plot.color import constant_color_dict


def main():
    function_family_type = "tactile"  # 'standard' or 'ANN' or 'tactile'
    path_to_nkg_files = Path(
        Path(__file__).parent.parent.parent, "kymata-core-data", "output"
    )

    # template invoker for printing out expression set .nkgs

    if function_family_type == "standard":
        expression_data = load_expression_set(
            Path(path_to_nkg_files, "combined_TVL_gridsearch.nkg")
        )

        fig = expression_plot(
            expression_data,
            color={
                "IL": "#b11e34",
                "IL1": "#a201e9",
                "IL2": "#a201e9",
                "IL3": "#a201e9",
                "IL4": "#a201e9",
                "IL5": "#a201e9",
                "IL6": "#a201e9",
                "IL7": "#a201e9",
                "IL8": "#a201e9",
                "IL9": "#a201e9",
                "STL": "#d388b5",
            },
            minimap="standard",
            ylim=-200,
        )

        fig.savefig(Path(path_to_nkg_files, "expression_plot.png"))

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

    elif function_family_type == "tactile":

        # expression_data = load_expression_set(
        #     Path(path_to_nkg_files, 'new_fwd', 'two_reps_without_11_thumb', "all_tactile.nkg")
        # )

        expression_data = load_expression_set(
            Path(path_to_nkg_files, 'all_participants', 'raw', "all_tactile.nkg")
        )  

        name = expression_data.transforms

        # import ipdb; ipdb.set_trace()

        fig_1 = expression_plot(
            expression_data,
            # xlims=(-200, 800),
            # minimap = 'large',
            # show_legend=False,
            color=constant_color_dict(['LH_PC_1', 'RH_PC_1'], color="red")
            | constant_color_dict(['LH_PC_2', 'RH_PC_2'], color="blue")
            | constant_color_dict(['LH_PC_3', 'RH_PC_3'], color="green")
            | constant_color_dict(['LH_PC_4', 'RH_PC_4'], color="yellow")
            | constant_color_dict(['LH_PC_5', 'RH_PC_5'], color="orange"),
            show_only=[i for i in name if 'PC' in i],
            title = 'PC',
        )

        fig_2 = expression_plot(
            expression_data,
            # xlims=(-200, 800),
            # minimap = 'large',
            # show_legend=False,
            color=constant_color_dict(['LH_SA_1', 'RH_SA_1'], color="red")
            | constant_color_dict(['LH_SA_2', 'RH_SA_2'], color="blue")
            | constant_color_dict(['LH_SA_3', 'RH_SA_3'], color="green")
            | constant_color_dict(['LH_SA_4', 'RH_SA_4'], color="yellow")
            | constant_color_dict(['LH_SA_5', 'RH_SA_5'], color="orange"),
            show_only=[i for i in name if 'SA' in i],
            title = 'SA',
        )

        fig_3 = expression_plot(
            expression_data,
            # xlims=(-200, 800),
            minimap = 'large',
            # minimap_view='dorsal',
            # minimap_surface = 'pial',
            # show_legend=False,
            color=constant_color_dict(['LH_PC_1', 'RH_PC_1'], color="red")
            | constant_color_dict(['LH_PC_2', 'RH_PC_2'], color="blue")
            | constant_color_dict(['LH_PC_3', 'RH_PC_3'], color="green")
            | constant_color_dict(['LH_PC_4', 'RH_PC_4'], color="yellow")
            | constant_color_dict(['LH_PC_5', 'RH_PC_5'], color="orange"),
            show_only=[i for i in name if 'PC' in i],
            title = 'PC',
        )

        fig_4 = expression_plot(
            expression_data,
            # xlims=(-200, 800),
            minimap = 'large',
            # minimap_view='dorsal',
            # minimap_surface = 'pial',
            # show_legend=False,
            color=constant_color_dict(['LH_SA_1', 'RH_SA_1'], color="red")
            | constant_color_dict(['LH_SA_2', 'RH_SA_2'], color="blue")
            | constant_color_dict(['LH_SA_3', 'RH_SA_3'], color="green")
            | constant_color_dict(['LH_SA_4', 'RH_SA_4'], color="yellow")
            | constant_color_dict(['LH_SA_5', 'RH_SA_5'], color="orange"),
            show_only=[i for i in name if 'SA' in i],
            title = 'SA',
        )

        fig_1.savefig(Path(path_to_nkg_files, 'all_participants', 'plots', "PC.png"))
        fig_2.savefig(Path(path_to_nkg_files, 'all_participants', 'plots', "SA.png"))
        fig_3.savefig(Path(path_to_nkg_files, 'all_participants', 'plots', "PC_brain.png"))
        fig_4.savefig(Path(path_to_nkg_files, 'all_participants', 'plots', "SA_brain.png"))



        # for i in range(1,6):

        #     expression_data_Vib_detect = expression_data[f'LH_PC_{i}', f'RH_PC_{i}']
        #     expression_data_Vert_disp = expression_data[f'LH_SA_{i}', f'RH_SA_{i}']

        #     fig_1 = expression_plot(
        #         # expression_data,
        #         expression_data_Vib_detect,
        #         # xlims=(-200, 800),
        #         # minimap = 'large',
        #         # show_legend=False,
        #         color=constant_color_dict(['LH_PC_1', 'RH_PC_1'], color="red")
        #         | constant_color_dict(['LH_PC_2', 'RH_PC_2'], color="blue")
        #         | constant_color_dict(['LH_PC_3', 'RH_PC_3'], color="green")
        #         | constant_color_dict(['LH_PC_4', 'RH_PC_4'], color="yellow")
        #         | constant_color_dict(['LH_PC_5', 'RH_PC_5'], color="orange"),
        #         show_only=[i for i in name if 'SquareVib' in i],
        #         title = 'Vibration detection',
        #     )

        #     fig_2 = expression_plot(
        #         # expression_data,
        #         expression_data_Vert_disp,
        #         # xlims=(-200, 800),
        #         # minimap = 'large',
        #         # show_legend=False,
        #         color=constant_color_dict(['LH_SA_1', 'RH_SA_1'], color="red")
        #         | constant_color_dict(['LH_SA_2', 'RH_SA_2'], color="blue")
        #         | constant_color_dict(['LH_SA_3', 'RH_SA_3'], color="green")
        #         | constant_color_dict(['LH_SA_4', 'RH_SA_4'], color="yellow")
        #         | constant_color_dict(['LH_SA_5', 'RH_SA_5'], color="orange"),
        #         show_only=[i for i in name if 'slowfluct' in i],
        #         title = 'Vertical displacement',
        #     )

        #     fig_3 = expression_plot(
        #         # expression_data,
        #         expression_data_Vib_detect,
        #         # xlims=(-200, 800),
        #         minimap = 'large',
        #         # minimap_view='dorsal',
        #         # minimap_surface = 'pial',
        #         # show_legend=False,
        #         color=constant_color_dict(['LH_PC_1', 'RH_PC_1'], color="red")
        #         | constant_color_dict(['LH_PC_2', 'RH_PC_2'], color="blue")
        #         | constant_color_dict(['LH_PC_3', 'RH_PC_3'], color="green")
        #         | constant_color_dict(['LH_PC_4', 'RH_PC_4'], color="yellow")
        #         | constant_color_dict(['LH_PC_5', 'RH_PC_5'], color="orange"),
        #         show_only=[i for i in name if 'SquareVib' in i],
        #         title = 'Vibration detection',
        #     )

        #     fig_4 = expression_plot(
        #         # expression_data,
        #         expression_data_Vert_disp,
        #         # xlims=(-200, 800),
        #         minimap = 'large',
        #         # minimap_view='dorsal',
        #         # minimap_surface = 'pial',
        #         # show_legend=False,
        #         color=constant_color_dict(['LH_SA_1', 'RH_SA_1'], color="red")
        #         | constant_color_dict(['LH_SA_2', 'RH_SA_2'], color="blue")
        #         | constant_color_dict(['LH_SA_3', 'RH_SA_3'], color="green")
        #         | constant_color_dict(['LH_SA_4', 'RH_SA_4'], color="yellow")
        #         | constant_color_dict(['LH_SA_5', 'RH_SA_5'], color="orange"),
        #         show_only=[i for i in name if 'slowfluct' in i],
        #         title = 'Vertical displacement',
        #     )

        #     fig_1.savefig(Path(path_to_nkg_files, 'new_fwd', 'two_reps_without_11_thumb', 'plots', f"Vib_detect_{i}.png"))
        #     fig_2.savefig(Path(path_to_nkg_files, 'new_fwd', 'two_reps_without_11_thumb', 'plots', f"Vert_disp_{i}.png"))
        #     fig_3.savefig(Path(path_to_nkg_files, 'new_fwd', 'two_reps_without_11_thumb', 'plots', f"Vib_detect_brain_{i}.png"))
        #     fig_4.savefig(Path(path_to_nkg_files, 'new_fwd', 'two_reps_without_11_thumb', 'plots', f"Vert_disp_brain_{i}.png"))



if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
