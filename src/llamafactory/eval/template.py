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

from dataclasses import dataclass

from ..data import Role
from ..extras.constants import CHOICES


@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str

    def _parse_example(self, example: dict[str, str]) -> tuple[str, str]:
        r"""Parse eval example.

        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response).
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
            self, target_data: dict[str, str], support_set: list[dict[str, str]], subject_name: str
    ) -> list[dict[str, str]]:
        r"""Convert dataset examples to messages."""
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates: dict[str, "EvalTemplate"] = {}


def _register_eval_template(name: str, system: str, choice: str, answer: str) -> None:
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer)


def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, f"Template {name} does not exist."
    return eval_template


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer:",
)

_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
)
