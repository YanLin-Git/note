
1. 多模态 (文-->图)
    1. stable-diffusion
        > Hugging Face 地址: https://huggingface.co/stabilityai/stable-diffusion-3-medium

2. 多模态 (图-->文)
    > 参考链接: https://zhuanlan.zhihu.com/p/653902791

    |模型|文本编码|图像编码|图文结合部分|训练模块|
    |---|---|---|---|---|
    |CLIP|BERT|ViT|无|BERT+ViT|
    |BLIP|BERT|ViT|通过注意力机制来交互|BERT+ViT|
    |BLIP2|LLM|CLIP-ViT|Q-Former|Q-Former|
    |VisualGLM|ChatGLM-6B|~~未查到~~|Q-Former|~~未查到~~|
    |llava|LLaMA/Vicuna|CLIP-ViT|映射矩阵W|映射矩阵W+LLaMA|
    |xtuner/llava-llama-3-8b-v1_1-gguf|meta-llama/Meta-Llama-3-8B-Instruct|CLIP-ViT-Large-patch14-336 |同上|同上|
    |llava-hf/llava-1.5-13b-hf|~~未查到~~|~~未查到~~|同上|同上|

