
python test.py ./configs/EgoPER/coffee_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/coffee_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task coffee --dirname coffee_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -as
python test_ed.py ./configs/EgoPER/coffee_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/coffee_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task coffee --dirname coffee_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -ed

python test.py ./configs/EgoPER/tea_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/tea_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task tea --dirname tea_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -as
python test_ed.py ./configs/EgoPER/tea_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/tea_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task tea --dirname tea_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -ed

python test.py ./configs/EgoPER/quesadilla_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/quesadilla_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task quesadilla --dirname quesadilla_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -as
python test_ed.py ./configs/EgoPER/quesadilla_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/quesadilla_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task quesadilla --dirname quesadilla_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -ed

python test.py ./configs/EgoPER/oatmeal_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/oatmeal_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task oatmeal --dirname oatmeal_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -as
python test_ed.py ./configs/EgoPER/oatmeal_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/oatmeal_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task oatmeal --dirname oatmeal_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -ed

python test.py ./configs/EgoPER/pinwheels_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/pinwheels_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task pinwheels --dirname pinwheels_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -as
python test_ed.py ./configs/EgoPER/pinwheels_aod_b1_n2_smooth_causal_gmm.yaml ./ckpt/EgoPER/pinwheels_aod_b1_n2_smooth_causal_gmm_final
python metric_vis_egoper.py --task pinwheels --dirname pinwheels_aod_b1_n2_smooth_causal_gmm_final/ --dirs EgoPER -ed



# 在线测评脚本
python metric_vis_egoper.py --dataset EgoPER --dirs EgoPER_taskgraph_pcml --dirname coffee_aod_b1_n2_smooth_causal_gmm_pcml_final --task coffee -ed --online >> log/coffee_online/metric.log