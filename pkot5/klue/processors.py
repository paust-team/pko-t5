import copy
from abc import abstractmethod, ABCMeta
from typing import Tuple, List, Dict

import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
from scipy.stats import pearsonr
import datasets


class KlueProcessor(metaclass=ABCMeta):
    def __init__(self, tokenizer, data):
        self.data = data
        self.tokenizer = tokenizer

    def process(self, split_name):
        input_texts, target_texts = self._process_t2t(self.data[split_name])
        input_ids = self.tokenizer(input_texts, add_special_tokens=True).input_ids
        label_ids = self.tokenizer(target_texts, add_special_tokens=True).input_ids

        assert len(input_ids) == len(self.data[split_name]), f"{len(input_ids)} == {len(self.data[split_name])}"
        assert len(label_ids) == len(self.data[split_name])

        return [
            dict(input_ids=input_ids[i], label_ids=label_ids[i], _input_text=input_texts[i], _target_text=target_texts[i], **row)
            for i, row in enumerate(self.data[split_name])
            if input_texts[i] != "" and target_texts[i] != ""
        ]

    def compute_metrics(self, output_ids, entries, **kwargs):
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return self._compute_metrics(output_texts, entries)

    @abstractmethod
    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        pass

    @abstractmethod
    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        pass


class KlueYnatProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, data=datasets.load_dataset('klue', name='ynat'))
        self.label2texts = {
            0: "IT과화",
            1: "경제",
            2: "사회",
            3: "생활문화",
            4: "세계",
            5: "스표츠",
            6: "정치"
        }

    @property
    def task(self):
        return 'ynat'

    def _process_t2t(self, dataset):
        input_texts, target_texts = [], []
        for row in dataset:
            row = dict(row)
            label = int(row['label'])
            input_texts.append(f"klue ynat title: {row['title']}")
            assert label in self.label2texts
            target_texts.append(self.label2texts[label])

        return input_texts, target_texts

    def _compute_metrics(self, output_texts, entries):
        text2labels = {v: k for k, v in self.label2texts.items()}
        pred_ids = [text2labels.get(t.strip(), -1) for t in output_texts]
        label_ids = [int(t['label']) for t in entries]

        return {'f1': f1_score(label_ids, pred_ids, average='macro')}


class KlueSTSProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='sts')
        )

    @property
    def task(self):
        return 'sts'

    def _process_t2t(self, dataset):
        input_texts, target_texts = [], []
        for row in dataset:
            label = float(row['labels']['label'])
            input_texts.append(f"klue sts sentence1: {row['sentence1']} sentence2: {row['sentence2']}")
            target_texts.append(f"{label}")

        return input_texts, target_texts

    def _compute_metrics(self, pred_texts, entries):
        pred_ids = []
        pred_scores = []
        for pred in pred_texts:
            try:
                pred_score = float(pred)
                pred_ids.append(1 if pred_score >= 3.0 else 0)
                pred_scores.append(pred_score)
            except ValueError:
                pred_ids.append(0)
                pred_scores.append(0.0)

        target_scores = [float(t['labels']['label']) for t in entries]
        target_ids = [1 if t >= 3.0 else 0 for t in target_scores]

        r, _ = pearsonr(pred_scores, target_scores)
        f1 = f1_score(target_ids, pred_ids)

        return {"pearsonr": r, "f1": f1}


class KlueNLIProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(data=datasets.load_dataset('klue', name='nli'), tokenizer=tokenizer)
        self.label2texts = {
            0: 'entailment',
            1: 'neutral',
            2: 'contradiction',
        }

    @property
    def task(self):
        return 'nli'

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        input_texts, target_texts = [], []
        for row in dataset:
            label = row['label']
            input_texts.append(f"klue nli premise: {row['premise']} hypothesis: {row['hypothesis']}")
            assert label in self.label2texts
            target_texts.append(self.label2texts[label])

        return input_texts, target_texts

    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        text2labels = {v: k for k, v in self.label2texts.items()}
        pred_ids = [text2labels.get(t.strip(), -1) for t in output_texts]
        target_ids = [t['label'] for t in entries]

        return {"acc": accuracy_score(target_ids, pred_ids)}


class KlueNERProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='ner')
        )

    @property
    def task(self):
        return 'ner'

    def _get_entities(self, row):
        entities = []
        entity_text = ""
        entity_label = ""
        state = "NONE"
        for ch in row['sentence']:
            if state == "NONE" and ch == "<":
                state = "PROC_1"
                entity_text = ""
                entity_label = ""
            elif state == "PROC_1" and ch == ":":
                state = "PROC_2"
            elif state == "PROC_1":
                assert ch not in "<:>"
                entity_text += ch
            elif state == "PROC_2" and ch == ">":
                state = "NONE"
                entities.append((entity_text, entity_label))
            elif state == "PROC_2":
                assert ch not in "<:>", f"ch: {ch}, sentence: {row['sentence']}"
                entity_label += ch
        return entities

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        input_texts, target_texts = [], []
        for row in dataset:
            sentence = ''.join(row['tokens'])

            try:
                entities = self._get_entities(row)
                input_texts.append(f"klue ner sentence: {sentence}")
                target_text = ' '.join([f"{text} [{label}]" for text, label in entities])
                target_texts.append(target_text)
            except AssertionError:
                input_texts.append("")
                target_texts.append("")

        return input_texts, target_texts

    def _do_recovery(self, text):
        state = 0
        entity_text = ""
        entity_label = ""
        for ch in text:
            if state == 0 and ch != " ":
                state = 1
                entity_text += str(ch)
            elif state == 1 and ch == "[":
                state = 2
            elif state == 1:
                entity_text += str(ch)
            elif state == 2 and ch == "]":
                state = 0
                assert entity_text != ""
                assert entity_label != ""
                yield entity_text.strip(), entity_label
                entity_text = ""
                entity_label = ""
            elif state == 2:
                entity_label += str(ch)

    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        num_correct = 0
        num_targets = 0
        num_preds = 0
        for i, (pred_text, row) in enumerate(zip(output_texts, entries)):
            preds = list(self._do_recovery(pred_text))
            targets = self._get_entities(row)

            num_targets += len(targets)
            num_preds += len(preds)

            preds = {text: label for text, label in preds}
            for text, label in targets:
                if text in preds:
                    pred_label = preds[text]
                    if label == pred_label:
                        num_correct += 1

        prec = num_correct / num_preds
        recall = num_correct / num_targets
        f1 = (2 * prec * recall) / (prec + recall + 1e-8)
        return {"f1": f1}


class KlueREProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='re')
        )
        self.label_keys = [
            "no_relation",
            "org:dissolved",
            "org:founded",
            "org:place_of_headquarters",
            "org:alternate_names",
            "org:member_of",
            "org:members",
            "org:political/religious_affiliation",
            "org:product",
            "org:founded_by",
            "org:top_members/employees",
            "org:number_of_employees/members",

            "per:date_of_birth",
            "per:date_of_death",
            "per:place_of_birth",
            "per:place_of_death",
            "per:place_of_residence",
            "per:origin",
            "per:employee_of",
            "per:schools_attended",
            "per:alternate_names",
            "per:parents",
            "per:children",
            "per:siblings",
            "per:spouse",
            "per:other_family",

            "per:colleagues",
            "per:product",
            "per:religion",
            "per:title",
        ]
        labels = {label: idx for idx, label in enumerate(self.label_keys)}
        self.labels = labels

    @property
    def task(self):
        return 're'

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        labels = self.label_keys
        input_texts, target_texts = [], []
        for row in dataset:
            sentence = row['sentence']
            subject = row['subject_entity']
            obj = row['object_entity']
            text = f"klue re sentence: {sentence} subject: {subject['word']}({subject['type']}) object: {obj['word']}({obj['type']})"
            label = row['label']
            input_texts.append(text)
            assert label < len(labels), f"label: {label}"
            target_texts.append(labels[label])

        return input_texts, target_texts

    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        text2labels = {v: k for k, v in enumerate(self.labels)}
        pred_ids, target_ids = [], []
        for pred_text, row in zip(output_texts, entries):
            pred_id = text2labels.get(pred_text.strip(), -1)
            target_id = row['label']
            if target_id != 0:
                pred_ids.append(pred_id)
                target_ids.append(target_id)

        micro_f1 = f1_score(target_ids, pred_ids, average='micro')
        return {"micro_f1": micro_f1}


class KlueDPProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='dp')
        )

    @property
    def task(self):
        return 'dp'

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        input_texts, target_texts = [], []
        for row in dataset:
            assert '[' not in row['sentence'] and ']' not in row['sentence']
            assert len(row['word_form']) == len(list(row['sentence'].split(' '))), f"{len(row['word_form'])} == {len(list(row['sentence'].split(' ')))}"
            inputs, targets = [], []
            for word, tag, deprel in zip(row['word_form'], row['pos'], row['deprel']):
                inputs.append(f"{word}({tag})")
                targets.append(f"{word}[{deprel}]")
            input_texts.append("klue dp: " + " ".join(inputs))
            target_texts.append(" ".join(targets))

        return input_texts, target_texts

    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        corrections = 0
        num_preds = 0
        num_targets = 0
        for i, (pred_text, row) in enumerate(zip(output_texts, entries)):
            preds = list(pred_text.split(' '))
            targets = list(row['_target_text'].split(' '))
            num_preds += len(preds)
            num_targets += len(targets)
            for i in range(min(len(preds), len(targets))):
                pred = preds[i]
                target = targets[i]
                label_p = pred[pred.find('[')+1:pred.find(']')]
                label_t = target[target.find('[')+1:target.find(']')]
                text_p = pred[:pred.find('[')]
                text_t = target[:target.find('[')]

                if label_p == label_t and text_p == text_t:
                    corrections += 1


        precision = corrections / num_preds
        recall = corrections / num_targets
        las = (2. * precision * recall) / (precision + recall)
        return {'las': las}


class KlueMRCProcessorWithSliding(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='mrc')
        )
        self.squad_metric = datasets.load_metric('squad')

    @property
    def task(self):
        return 'mrc'

    def process(self, split_name):
        guids, input_texts1, input_texts2, target_texts = self._process_t2t(self.data[split_name])
        all_input_ids1 = self.tokenizer(input_texts1, add_special_tokens=False).input_ids
        all_input_ids2 = self.tokenizer(input_texts2, add_special_tokens=False).input_ids
        all_label_ids = self.tokenizer(target_texts, add_special_tokens=True).input_ids

        assert len(all_input_ids1) == len(self.data[split_name]), f"{len(all_input_ids1)} == {len(self.data[split_name])}"
        assert len(all_label_ids) == len(self.data[split_name])

        features = []
        for guid, ids1, ids2, label_ids, record in zip(guids, all_input_ids1, all_input_ids2, all_label_ids, self.data[split_name]):
            context_max_len = 512 - len(ids1) - 1
            sub_context_ids = [ids2[begin:begin + context_max_len] for begin in range(0, len(ids2), 128)]
            sub_contexts = self.tokenizer.batch_decode(sub_context_ids, skip_special_tokens=True)
            answer = self.tokenizer.batch_decode([label_ids], skip_special_tokens=True)[0]
            assert any([answer in c for c in sub_contexts])
            for ctx, ctx_ids in zip(sub_contexts, sub_context_ids):
                input_ids = ids1 + ctx_ids + [self.tokenizer.eos_token_id]
                if answer not in ctx:
                    label_ids = [self.tokenizer.eos_token_id]
                features.append(dict(input_ids=input_ids, label_ids=label_ids, **record))

        return features

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str], List[str], List[str]]:
        guids, input_texts1, input_texts2, target_texts = [], [], [], []
        for row in dataset:
            guid = row['guid']
            context = row['context']
            question = row['question']
            answers = row['answers']['text']
            answers = [answer.replace("&amp;", "&") for answer in answers]
            row['answers']['text'] = answers
            assert len(answers) > 0

            input_texts1.append(f"klue mrc question: {question} context: ")
            input_texts2.append(context)
            target_texts.append(f"{answers[-1]}")
            guids.append(guid)
        assert len(input_texts1) == len(dataset)
        assert len(target_texts) == len(dataset)
        return guids, input_texts1, input_texts2, target_texts

    def compute_metrics(self, output_ids, entries, output_scores=None, **kwargs):
        assert output_scores is not None
        output_texts = [
            self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        return self._compute_metrics(output_texts, output_scores, entries)

    @torch.no_grad()
    def _compute_metrics(self, output_texts: List[List[str]], output_scores: List[List[float]], entries: List[Dict[str, any]]) -> Dict[str, float]:
        features = {}
        for i, feature in enumerate(entries):
            guid = feature['guid']
            if guid not in features:
                feat = copy.deepcopy(feature)
                feat['output_texts'] = output_texts[i]
                feat['output_scores'] = output_scores[i]
                features[guid] = feat
            else:
                feat = features[guid]
                feat['output_texts'] += output_texts[i]
                feat['output_scores'] += output_scores[i]

        predictions = []
        references = []
        for guid, feature in features.items():
            answers = []
            context = feature['context']
            for output_text, output_score in zip(feature['output_texts'], feature['output_scores']):
                if len(output_text.strip()) > 0 and output_text in context:
                    answers.append({'text': output_text, 'score': output_score})

            best_answer = max(answers, key=lambda t: t['score'])['text'] if len(answers) > 0 else ""
            predictions.append({'prediction_text': best_answer, 'id': guid})
            references.append({'answers': feature['answers'], 'id': guid})

        results = self.squad_metric.compute(predictions=predictions, references=references)

        return {"exact_match": results['exact_match'], 'f1': results['f1']}


class KlueMRCProcessor(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='mrc')
        )
        self.squad_metric = datasets.load_metric('squad')

    @property
    def task(self):
        return 'mrc'

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        input_texts, target_texts = [], []
        for row in dataset:
            context = row['context']
            question = row['question']
            answers = row['answers']['text']
            answers = [answer.replace("&amp;", "&") for answer in answers]
            row['answers']['text'] = answers
            assert len(answers) > 0

            input_texts.append(f"klue mrc question: {question} context: {context}")
            target_texts.append(f"{answers[-1]}")
        assert len(input_texts) == len(dataset)
        assert len(target_texts) == len(dataset)
        return input_texts, target_texts

    @torch.no_grad()
    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        references, predictions = [], []
        for output_text, row in zip(output_texts, entries):
            predictions.append({'prediction_text': output_text, 'id': row['guid']})
            references.append({'answers': row['answers'], 'id': row['guid']})
        results = self.squad_metric.compute(predictions=predictions, references=references)

        return {"exact_match": results['exact_match'], 'f1': results['f1']}


class KlueMRCProcessorWithTitle(KlueProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            data=datasets.load_dataset('klue', name='mrc')
        )
        self.squad_metric = datasets.load_metric('squad')

    @property
    def task(self):
        return 'mrc'

    def _process_t2t(self, dataset) -> Tuple[List[str], List[str]]:
        input_texts, target_texts = [], []
        for row in dataset:
            context = row['context']
            question = row['question']
            answers = row['answers']['text']
            title = row['title'].strip()
            is_impossible = row['is_impossible']
            answers = [answer.replace("&amp;", "&") for answer in answers]
            row['answers']['text'] = answers
            assert len(answers) > 0

            input_texts.append(f"klue mrc question: {question} title: {title} context: {context}")
            target_texts.append(f"{answers[-1]}" if not is_impossible else "")
        assert len(input_texts) == len(dataset)
        assert len(target_texts) == len(dataset)
        return input_texts, target_texts

    @torch.no_grad()
    def _compute_metrics(self, output_texts: List[str], entries: List[Dict[str, any]]) -> Dict[str, float]:
        references, predictions = [], []
        for output_text, row in zip(output_texts, entries):
            predictions.append({'prediction_text': output_text, 'id': row['guid']})
            references.append({'answers': row['answers'], 'id': row['guid']})
        results = self.squad_metric.compute(predictions=predictions, references=references)

        return {"exact_match": results['exact_match'], 'f1': results['f1']}


KLUE_PROCESSORS = {
    'ynat': KlueYnatProcessor,
    'sts': KlueSTSProcessor,
    'nli': KlueNLIProcessor,
    'ner': KlueNERProcessor,
    're': KlueREProcessor,
    'dp': KlueDPProcessor,
    'mrc': KlueMRCProcessor,  # KlueMRCProcessorWithTitle
}


class Text2TextDataset(torch.utils.data.Dataset):
    def __init__(self, entries, max_length):
        super().__init__()

        self.entries = entries
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        entry = self.entries[item]
        input_ids = entry['input_ids'][:self.max_length]
        return {
            'input_ids': input_ids,
            'attention_mask': [1] * len(input_ids),
            'labels': entry['label_ids'],
        }
