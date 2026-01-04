import yaml
from pathlib import Path
from collections import defaultdict

# Mappa delle classi dal file YAML
class_names = {
    0: "unknown_tool",
    1: "dissector",
    2: "scissors",
    3: "suction",
    4: "grasper 3",
    5: "harmonic",
    6: "grasper",
    7: "bipolar",
    8: "grasper 2",
    9: "cautery (hook, spatula)",
    10: "ligasure",
    11: "stapler",
    12: "unknown_tti",
    13: "coagulation",
    14: "other",
    15: "retract and grab",
    16: "blunt dissection",
    17: "energy - sharp dissection",
    18: "staple",
    19: "retract and push",
    20: "cut - sharp dissection"
}


def load_class_names_from_yaml(yaml_path):
    """Carica i nomi delle classi dal file YAML"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'names' in data:
            return data['names']
    except Exception as e:
        print(f"Errore nel caricamento del YAML: {e}")
    return class_names


def count_classes_in_dataset(dataset_path):
    """
    Conta le occorrenze di ogni classe nel dataset YOLO
    """
    dataset_path = Path(dataset_path)
    class_counts = defaultdict(int)
    
    # Cartelle da processare
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = dataset_path / 'labels' / split
        
        if not labels_dir.exists():
            print(f"Cartella {labels_dir} non trovata, saltando...")
            continue
        
        print(f"\nScansionando {split}...")
        
        # Processa tutti i file di label
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Errore nel leggere {label_file}: {e}")
                continue
            
            # Conta le classi
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                try:
                    class_id = int(parts[1])  # La classe è la seconda colonna (dopo la flag)
                    class_counts[class_id] += 1
                except ValueError:
                    continue
    
    return class_counts


def print_statistics(class_counts, class_names_dict):
    """Stampa le statistiche con i nomi delle classi"""
    print("\n" + "="*80)
    print(f"{'Classe ID':<12} {'Nome Classe':<35} {'Occorrenze':<15}")
    print("="*80)
    
    total = sum(class_counts.values())
    
    # Ordina per ID classe
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        name = class_names_dict.get(class_id, "Sconosciuto")
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{class_id:<12} {name:<35} {count:<10} ({percentage:>5.2f}%)")
    
    print("="*80)
    print(f"{'TOTALE':<12} {'':<35} {total:<10}")
    print("="*80)


def main():
    # Specifica i percorsi
    script_dir = Path(__file__).parent
    dataset_path = script_dir / 'dataset'
    yaml_path = script_dir / 'dataset copy.yaml'
    
    if not dataset_path.exists():
        print(f"Errore: il dataset non trovato in {dataset_path}")
        return
    
    print(f"Scansionando dataset in: {dataset_path}")
    
    # Carica i nomi delle classi dal YAML
    print(f"\nCaricando i nomi delle classi da: {yaml_path}")
    class_names_dict = load_class_names_from_yaml(yaml_path)
    
    # Conta le occorrenze
    class_counts = count_classes_in_dataset(dataset_path)
    
    # Stampa le statistiche
    print_statistics(class_counts, class_names_dict)
    
    # Stampa anche un riassunto
    print("\n" + "="*80)
    print("RIASSUNTO:")
    print("="*80)
    print(f"Numero di classi uniche presenti: {len(class_counts)}")
    print(f"Classe con più occorrenze: {max(class_counts, key=class_counts.get)} - "
          f"{class_names_dict.get(max(class_counts, key=class_counts.get))} "
          f"({class_counts[max(class_counts, key=class_counts.get)]} occorrenze)")
    print(f"Classe con meno occorrenze: {min(class_counts, key=class_counts.get)} - "
          f"{class_names_dict.get(min(class_counts, key=class_counts.get))} "
          f"({class_counts[min(class_counts, key=class_counts.get)]} occorrenze)")
    print("="*80)


if __name__ == '__main__':
    main()
