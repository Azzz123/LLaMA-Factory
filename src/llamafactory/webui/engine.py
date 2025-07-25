# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

from .chatter import WebChatModel
from .common import create_ds_config, get_time, load_config
from .locales import LOCALES
from .manager import Manager
from .runner import Runner

if TYPE_CHECKING:
    from gradio.components import Component


class Engine:
    r"""A general engine to control the behaviors of Web UI."""

    def __init__(self, demo_mode: bool = False, pure_chat: bool = False) -> None:
        self.demo_mode = demo_mode
        self.pure_chat = pure_chat
        self.manager = Manager()
        self.runner = Runner(self.manager, demo_mode)
        self.chatter = WebChatModel(self.manager, demo_mode, lazy_init=(not pure_chat))
        if not demo_mode:
            create_ds_config()

    def _update_component(self, input_dict: dict[str, dict[str, Any]]) -> dict["Component", "Component"]:
        r"""Update gradio components according to the (elem_id, properties) mapping."""
        output_dict: dict[Component, Component] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)

        return output_dict

    def resume(self):
        r"""Get the initial value of gradio components and restores training status if necessary."""
        user_config = load_config() if not self.demo_mode else {}  # do not use config in demo mode
        lang = user_config.get("lang") or "en"
        init_dict = {"top.lang": {"value": lang}, "infer.chat_box": {"visible": self.chatter.loaded}}

        if not self.pure_chat:
            current_time = get_time()
            hub_name = user_config.get("hub_name") or "huggingface"
            init_dict["top.hub_name"] = {"value": hub_name}
            init_dict["train.current_time"] = {"value": current_time}
            init_dict["train.output_dir"] = {"value": f"train_{current_time}"}
            init_dict["train.config_path"] = {"value": f"{current_time}.yaml"}
            init_dict["eval.output_dir"] = {"value": f"eval_{current_time}"}
            init_dict["infer.mm_box"] = {"visible": False}

            if user_config.get("last_model", None):
                init_dict["top.model_name"] = {"value": user_config["last_model"]}

        yield self._update_component(init_dict)

        if self.runner.running and not self.demo_mode and not self.pure_chat:
            yield {elem: elem.__class__(value=value) for elem, value in self.runner.running_data.items()}
            if self.runner.do_train:
                yield self._update_component({"train.resume_btn": {"value": True}})
            else:
                yield self._update_component({"eval.resume_btn": {"value": True}})

    def change_lang(self, lang: str):
        r"""Update the displayed language of gradio components."""
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
