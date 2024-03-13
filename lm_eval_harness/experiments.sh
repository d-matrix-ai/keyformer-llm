## =================== MPT-7B ====================
#bash run.sh <task> <shots> <model name> <model path> <model type> <dtype> <1/0 (1 for Keyformer)> <KV cache percentage> <recent window percentage>
# ==> Full Attention
bash run.sh openbookqa 0 mosaicml/mpt-7b ./MPT-7B mosaicml-mpt-7b float16 0 60 30
# ==> Keyformer
bash run.sh openbookqa 0 mosaicml/mpt-7b ./MPT-7B mosaicml-mpt-7b float16 1 60 30



