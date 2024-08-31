from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPPreTrainedModel, CLIPVisionModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling

class FieldsVisionConfig(PretrainedConfig):
    # modified from CLIPVisionConfig
    model_type = "fields_vision_model"
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size: Union[int, Tuple[int, int]] = (128, 128),
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        input_padding: Union[int, Tuple[int, int]] = (0, 0),
        output_hidden_state=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.input_padding = input_padding
        self.output_hidden_state = output_hidden_state

class FieldsEmbeddings(nn.Module):
    # modified from CLIPVisionEmbddings in:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
    def __init__(self, config: FieldsVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        self.input_padding = config.input_padding
        if isinstance(self.input_padding, int):
            self.input_padding = (self.input_padding, self.input_padding)

        self.patch_embedding = PatchEmbed(
            height=self.image_size[0],
            width=self.image_size[1],
            patch_size=self.patch_size,
            in_channels=config.num_channels,
            embed_dim=self.embed_dim,
            bias=True,
            pos_embed_type=None,
        )

        self.sensing_array_patch_embedding = PatchEmbed(
            height=self.image_size[0],
            width=self.image_size[1],
            patch_size=self.patch_size,
            in_channels=1,
            embed_dim=2*self.embed_dim,
            bias=True,
            pos_embed_type=None,
        )


        padded_inputs = tuple(a + b for a, b in zip(self.input_padding, self.image_size))
        num_patch_height = padded_inputs[0] // self.patch_size
        num_patch_width = padded_inputs[1] // self.patch_size

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, (num_patch_height, num_patch_width), base_size=num_patch_height)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self._mod_init_weights()

    def _mod_init_weights(self):
        # Since we modify the class name, we need to handle the weight initialization ourselves
        factor = self.config.initializer_factor
        nn.init.normal_(self.patch_embedding.proj.weight, std=self.config.initializer_range * factor)
        nn.init.constant_(self.patch_embedding.proj.bias, 0.0)
        nn.init.normal_(self.sensing_array_patch_embedding.proj.weight, std=self.config.initializer_range * factor)
        nn.init.constant_(self.sensing_array_patch_embedding.proj.bias, 0.0)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.proj.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values[:,:-1].to(dtype=target_dtype))  # shape = [*, num_patches, embed_dim]
        sensing_array_patch_embeds = self.sensing_array_patch_embedding(pixel_values[:,[-1]].to(dtype=target_dtype))
        scale, shift = torch.chunk(sensing_array_patch_embeds, 2, dim=-1)
        patch_embeds = patch_embeds * (1 + scale) + shift

        embeddings = patch_embeds + self.pos_embed
        return embeddings

class FieldsVisionTransformer(nn.Module):
    # modified from CLIPVisionTransformer in:
    def __init__(self, config: FieldsVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = FieldsEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        if not config.output_hidden_state:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        if not self.config.output_hidden_state:
            pooled_output = last_hidden_state.mean(dim=1)
            pooled_output = self.post_layernorm(pooled_output)
        else:
            pooled_output = None

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class FieldsVisionModelWithProjection(CLIPPreTrainedModel):
    # modified from: CLIPVisionModelWithProjection
    config_class = FieldsVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: FieldsVisionConfig):
        super().__init__(config)

        self.vision_model = FieldsVisionTransformer(config)
        self.config = config

        if not config.output_hidden_state:
            self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
            self._mod_init_weights()

        # Initialize weights and apply final processing
        self.post_init()
    
    def _mod_init_weights(self):
        # Since we modify the class name, we need to handle the weight initialization ourselves
        nn.init.normal_(
            self.visual_projection.weight,
            std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.config.output_hidden_state:
            pooled_output = vision_outputs[1]  # pooled_output
            image_embeds = self.visual_projection(pooled_output)
        else:
            image_embeds = None

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )