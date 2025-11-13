# python test.py ./configs/CaptainCook4d/cook4d_b2_n2_gmm.yaml ./ckpt/CaptainCook4D/cook4d_b2_n2_gmm_final
# python metric_vis_holo.py  --dirname cook4d_b2_n2_gmm_final/ -as --action verb --dirs HoloAssist_taskgraph
python test_ed.py ./configs/CaptainCook4d/cook4d_b2_n2_gmm.yaml ./ckpt/CaptainCook4D/cook4d_b2_n2_gmm_final_1013 >> log/captaincook4d_1013/test.log
# python metric_vis_cook4d.py  --dirname cook4d_b2_n2_gmm_final/ -ed --dirs CaptainCook4D >> log/captaincook4d/metric.log
# CaptainCook4D新判断逻辑
python metric_vis_cook4d.py  --dirname cook4d_b2_n2_gmm_final_1013/ -ed --dirs CaptainCook4D >> log/captaincook4d_1013/metric.log
# python metric_vis_holo.py  --dirname holo_bgr1.0_noun_b1_n2_smooth_causal_gmm_final/ -ed --action noun --dirs HoloAssist_taskgraph >> log/holonoun/metric.log
# python metric_vis_cook4d.py --dirname cook4d_b2_n2_gmm_final/ --task 0 -ed -as --dirs CaptainCook4d >> log/captaincook4d/metric_cook4d.log