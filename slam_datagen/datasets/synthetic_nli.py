import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import random
import networkx as nx
from pathlib import Path
from typing import List, Any
import json
from collections import defaultdict


@dataclass
class DatasetSample:
    id: int # уникальный идентификатор примера
    premise: str # текст посылки (может состоять из нескольких предложений)
    hypothesis: str # текст гипотезы (одно предложение)
    gold_label: str # одна из меток entailment | neutral | contradiction
    split: str # train | eval | (опционально) test
    hop_difficulty: int # целое число из множества {0,1,2}
    distractor_difficulty: int # целое число из множества {1,2,3}
    background_knowledge: str # explicit | implicit
    world_seed: str # как сделать!? сид, использованный для генерации скрытого мира (для воспроизводимости) 
    templates_version: str # что это?
    ood_tag: str # режим разбиения iid | entity_heldout | taxonomy_heldout | relation_pattern_heldout


def build_synthetic_nli_dataset(cfg: DictConfig) -> list[DatasetSample]:
    W = HiddenWord(n_entities=cfg.n_entities, word_seed=cfg.random_seed)
    W.sample_hiden_world()
    samples = []
    
    distractor_difficulty = [item.value for item in cfg.distractor_difficulty_distribution]
    distractor_weights = [item.weight for item in cfg.distractor_difficulty_distribution]
    hop_difficulty = [item.value for item in cfg.hop_difficulty_distribution]
    hop_weights = [item.weight for item in cfg.hop_difficulty_distribution]

    # idk how to not hardcode that
    chosen_distractors = random.choices(distractor_difficulty, weights=distractor_weights, k=cfg.n_train)
    choosen_hop = random.choices(hop_difficulty, weights=hop_weights, k=cfg.n_train)
    for i in range(cfg.n_train):
        if choosen_hop[i] == 0:
            samples.append(W.generate_zero_hop_sample(distractor_difficulty=chosen_distractors[i], split="train"))
        elif choosen_hop[i] == 1:
            samples.append(W.generate_one_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="train"))
        elif choosen_hop[i] == 2:
            samples.append(W.generate_two_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="train"))

    chosen_distractors = random.choices(distractor_difficulty, weights=distractor_weights, k=cfg.n_eval)
    choosen_hop = random.choices(hop_difficulty, weights=hop_weights, k=cfg.n_eval)
    for i in range(cfg.n_eval):
        if choosen_hop[i] == 0:
            samples.append(W.generate_zero_hop_sample(distractor_difficulty=chosen_distractors[i], split="eval"))
        elif choosen_hop[i] == 1:
            samples.append(W.generate_one_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="eval"))
        elif choosen_hop[i] == 2:
            samples.append(W.generate_two_hop_sample(distractor_difficulty=chosen_distractors[i], background_knowledge="explicit", split="eval"))
    
    return samples

def write_synthetic_nli_dataset(
    samples: list[DatasetSample],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_serialize_sample(sample)) + "\n")

    return output_path

def _serialize_sample(sample: DatasetSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "premise": sample.premise,
        "hypothesis": sample.hypothesis,
        "gold_label": sample.gold_label,
        "split": sample.split,
        "hop_difficulty": sample.hop_difficulty,
        "distractor_difficulty": sample.distractor_difficulty,
        "background_knowledge": sample.background_knowledge,
        "world_seed": sample.world_seed,
        "templates_version": sample.templates_version,
        "ood_tag": sample.ood_tag,
    }

class HiddenWord():
    def __init__(self, n_entities, templates_version=0, word_seed=11, ood_tag="iid"):
        self.last_used_id = 0
        self.n_entities = n_entities
        self.word_seed = word_seed
        self.ood_tag = ood_tag # todo realise different tags
        self.templates_version = templates_version

        random.seed(self.word_seed)

    def _create_types_polytree(self, types: List[str]):
        tree = nx.random_labeled_tree(len(types))
        polytree = nx.DiGraph()
        polytree.add_nodes_from(tree)
        polytree.add_edges_from(nx.bfs_edges(tree, 0))

        random.shuffle(types)
        for i in range(len(types)):
            polytree.nodes[i]["type"] = types[i]

        return polytree

    def _create_types_polyforest(self, types: List[str], avg_types_in_tree=16) -> List[nx.DiGraph]:
        polyforest = []

        random.shuffle(types)
        cur_type = 0
        while len(types) - cur_type > 32:
            cnt_types = random.randint(avg_types_in_tree - 5, avg_types_in_tree + 5)
            polyforest.append(self._create_types_polytree(types[cur_type : cur_type + cnt_types]))
            cur_type += cnt_types
        else:
            polyforest.append(self._create_types_polytree(types[cur_type:]))
                
        return polyforest
    
    @staticmethod
    def _get_leaves(polytree: nx.DiGraph) -> List[str]:
        return [polytree.nodes[n]["type"] for n in polytree.nodes if polytree.out_degree(n) == 0]

    @staticmethod
    def _get_leaf_nodes(polytree: nx.DiGraph) -> List[int]:
        return [n for n in polytree.nodes if polytree.out_degree(n) == 0]
            
    def sample_hiden_world(self):
        colors = [
            "red", "green", "blue", "yellow", "orange",
            "purple", "pink", "brown", "black", "white",
            "gray", "cyan", "magenta", "lime", "maroon",
            "navy", "olive", "teal", "silver", "gold",
            "coral", "indigo", "violet", "turquoise", "beige"
        ]

        locations = [
            "park", "forest", "beach", "mountain", "river",
            "lake", "ocean", "desert", "island", "valley",
            "cave", "waterfall", "volcano", "glacier", "jungle",
            "savanna", "tundra", "swamp", "canyon", "prairie",
            "reef", "coast", "plateau", "meadow", "hill"
        ]

        shapes = [
            "circle", "square", "triangle", "rectangle", "oval",
            "pentagon", "hexagon", "octagon", "diamond", "star",
            "crescent", "heart", "arrow", "cross", "spiral",
            "cylinder", "cube", "sphere", "cone", "pyramid",
            "torus", "prism", "parallelogram", "trapezoid", "ellipse"
        ]

        animals = [
            "dog", "cat", "elephant", "lion", "tiger",
            "bear", "zebra", "giraffe", "monkey", "dolphin",
            "whale", "shark", "eagle", "snake", "frog",
            "rabbit", "horse", "cow", "pig", "sheep",
            "wolf", "fox", "red panda", "kangaroo", "penguin"
        ]

        types = [f"t{i}" for i in range(1000)]
        chosen_types = random.sample(types, 500)
 
        entities = [f"e{i}" for i in range(self.n_entities * 2)]
        chosen_entities = random.sample(entities, self.n_entities)

        self.zero_hop_dict = {
            "colors": dict(),
            "locations": dict(),
            "shapes": dict(),
            "animals": dict(),
        }
        temp_dict = dict()
        self.one_hop_dict = defaultdict(list)
        self.polyforest = self._create_types_polyforest(chosen_types)

        leaf_types = [leaf for tree in self.polyforest for leaf in self._get_leaves(tree)]
        print(leaf_types)
        self.leaf_types = leaf_types
        self.add_true_label_to_all_entities(self.zero_hop_dict["colors"],
                                            random.sample(colors, 13),
                                            chosen_entities[:self.n_entities//4])
        self.add_true_label_to_all_entities(self.zero_hop_dict["locations"],
                                            random.sample(locations, 13),
                                            chosen_entities[self.n_entities//4 : self.n_entities//2])
        self.add_true_label_to_all_entities(self.zero_hop_dict["shapes"],
                                            random.sample(shapes, 13),
                                            chosen_entities[self.n_entities//2 : 3 * self.n_entities//4])
        self.add_true_label_to_all_entities(self.zero_hop_dict["animals"],
                                            random.sample(animals, 13),
                                            chosen_entities[3 * self.n_entities//4:])
        self.add_true_label_to_all_entities(temp_dict,
                                            leaf_types,
                                            chosen_entities)

        for key, value in temp_dict.items():
            self.one_hop_dict[value].append(key)

    def add_true_label_to_all_entities(self, d, labels: List[str], entities: List[str]):
        for entity in entities:
            d[entity] = random.choice(labels)

    def generate_zero_hop_sample(self, distractor_difficulty, split) -> DatasetSample:
        print("zero")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])
        type = random.choice(list(self.zero_hop_dict.keys()))
        entity = random.choice(list(self.zero_hop_dict[type].keys()))
        chosen_entities = {entity}
        premise = []

        if gold_label == "entailment":
            premise.append(f"{entity} is {self.zero_hop_dict[type][entity]}")
            hypothesis = premise[0]
        elif gold_label == "neutral":
            premise.append(f"{entity} is {self.zero_hop_dict[type][entity]}")
            entity = random.choice(list(self.zero_hop_dict[type].keys()))
            while entity in chosen_entities:
                entity = random.choice(list(self.zero_hop_dict[type].keys()))
            hypothesis = f"{entity} is {self.zero_hop_dict[type][entity]}"
            chosen_entities.add(entity)
        elif gold_label == "contradiction":
            premise.append(f"{entity} is {self.zero_hop_dict[type][entity]}")
            false_label = random.choice(list(self.zero_hop_dict[type].values()))
            while false_label == self.zero_hop_dict[type][entity]:
                false_label = random.choice(list(self.zero_hop_dict[type].values()))
            hypothesis = f"{entity} is {false_label}"
            chosen_entities = {entity}
            
        while len(chosen_entities) != distractor_difficulty + 1:
            entity = random.choice(list(self.zero_hop_dict[type].keys()))
            if entity not in chosen_entities:
                chosen_entities.add(entity)
                premise.append(f"{entity} is {self.zero_hop_dict[type][entity]}")
        
        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 0, distractor_difficulty, "explicit", self.word_seed, self.templates_version, self.ood_tag)

        return sample

    def generate_one_hop_sample(self, distractor_difficulty, background_knowledge, split) -> DatasetSample:
        print("one")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])

        while True:
            chosen_type_tree = random.choice(self.polyforest)
            chosen_leaf = random.choice(self._get_leaf_nodes(chosen_type_tree))
            child_type = chosen_type_tree.nodes[chosen_leaf]["type"]

            entities = self.one_hop_dict.get(child_type)
            if entities:
                break

        entity = random.choice(entities)
        parent = next(iter(chosen_type_tree.predecessors(chosen_leaf)))
        parent_type = chosen_type_tree.nodes[parent]["type"]
        
        chosen_entities = {entity}
        premise = [f"{entity} is a {child_type}"]

        chosen_types = set()
        chosen_types.add(child_type)

        if background_knowledge == "explicit":
            premise.append(f"{child_type} is a subtype of {parent_type}")

        if gold_label == "entailment":
            hypothesis = f"{entity} is a {parent_type}"
            no_type = random.choice([t for t in self.leaf_types if t not in {child_type}])
            premise.append(f"No entity is both {child_type} and {no_type}")
            chosen_types.add(no_type)
        elif gold_label == "neutral":
            other_type = random.choice([t for t in self.leaf_types if t != child_type])
            no_type = random.choice([t for t in self.leaf_types if t not in {child_type, other_type}])
            hypothesis = f"{entity} is a {other_type}"
            premise.append(f"No entity is both {child_type} and {no_type}")
            chosen_types.add(other_type)
            chosen_types.add(no_type)
        elif gold_label == "contradiction":
            other_type = random.choice([t for t in self.leaf_types if t != child_type])
            hypothesis = f"{entity} is a {other_type}"
            premise.append(f"No entity is both {child_type} and {other_type}")
            chosen_types.add(other_type)

        while len(premise) != distractor_difficulty + 2:
            while True:
                other_chosen_type_tree = random.choice(self.polyforest)
                other_chosen_leaf = random.choice(self._get_leaf_nodes(other_chosen_type_tree))
                other_child_type = other_chosen_type_tree.nodes[other_chosen_leaf]["type"]
    
                entities = self.one_hop_dict.get(other_child_type)
                if entities and other_child_type not in chosen_types:
                    break
            
            entity = random.choice(entities)
            premise += [f"{entity} is a {other_child_type}"]
        
        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 1, distractor_difficulty, background_knowledge, self.word_seed, self.templates_version, self.ood_tag)

        return sample
    
    def generate_two_hop_sample(self, distractor_difficulty, background_knowledge, split) -> DatasetSample:
        print("two")
        gold_label = random.choice(["entailment", "neutral", "contradiction"])

        while True:
            chosen_type_tree = random.choice(self.polyforest)
            chosen_leaf = random.choice(self._get_leaf_nodes(chosen_type_tree))
            child_type = chosen_type_tree.nodes[chosen_leaf]["type"]

            entities = self.one_hop_dict.get(child_type)
            if not entities:
                continue

            parents = list(chosen_type_tree.predecessors(chosen_leaf))
            if not parents:
                continue
            parent = parents[0]

            grandparents = list(chosen_type_tree.predecessors(parent))
            if not grandparents:
                continue
            grandparent = grandparents[0]
            break

        entity = random.choice(entities)
        parent_type = chosen_type_tree.nodes[parent]["type"]
        grandparent_type = chosen_type_tree.nodes[grandparent]["type"]

        related_types = {grandparent_type} | {
            chosen_type_tree.nodes[n]["type"] for n in nx.descendants(chosen_type_tree, grandparent)
        }
        outside_leaves = [t for t in self.leaf_types if t not in related_types]

        chosen_entities = {entity}
        premise = [f"{entity} is a {child_type}"]

        if background_knowledge == "explicit":
            premise.append(f"{child_type} is a subtype of {parent_type}")
            premise.append(f"{parent_type} is a subtype of {grandparent_type}")

        if gold_label == "entailment":
            hypothesis = f"{entity} is a {grandparent_type}"
            no_type = random.choice(outside_leaves)
            premise.append(f"No entity is both {grandparent_type} and {no_type}")
        elif gold_label == "neutral":
            other_type = random.choice(outside_leaves)
            no_type = random.choice([t for t in outside_leaves if t != other_type])
            hypothesis = f"{entity} is a {other_type}"
            premise.append(f"No entity is both {grandparent_type} and {no_type}")
        elif gold_label == "contradiction":
            other_type = random.choice(outside_leaves)
            hypothesis = f"{entity} is a {other_type}"
            premise.append(f"No entity is both {grandparent_type} and {other_type}")

        start_len = len(premise)
        while len(premise) != start_len + distractor_difficulty:
            other_type = random.choice(self.leaf_types)
            other_entities = self.one_hop_dict.get(other_type)
            if not other_entities:
                continue
            other_entity = random.choice(other_entities)
            if other_entity in chosen_entities:
                continue
            chosen_entities.add(other_entity)
            premise.append(f"{other_entity} is a {other_type}")

        self.last_used_id += 1
        random.shuffle(premise)
        sample = DatasetSample(self.last_used_id, premise, hypothesis, gold_label, split, 2, distractor_difficulty, background_knowledge, self.word_seed, self.templates_version, self.ood_tag)

        return sample

if __name__ == "__main__":
    W = HiddenWord(1)
    W.sample_hiden_world()