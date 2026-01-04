import os
import shutil
from pathlib import Path

def create_class_mapping():
    """
    Crea una mappa che rinomina le classi in base alle modifiche:
    - Rimuove classi 0 e 12
    - Unisce classi 4, 8, 6 -> 6
    - Unisce classi 15, 19 -> 15
    - Rinomina tutte le altre classi nel nuovo ordine
    """
    # Classi da eliminare
    remove_classes = {0, 10, 12}
    
    # Classi da unire
    unite_to_6 = {4, 6, 8}  # Unire 4, 6, 8 -> 6
    unite_to_15 = {15, 19}   # Unire 15, 19 -> 15
    
    # Passo 1: Crea una mappa intermedia (classi originali -> classi rappresentanti o None)
    intermediate_mapping = {}
    
    for old_class in range(21):  # 0-20
        if old_class in remove_classes:
            intermediate_mapping[old_class] = None
        elif old_class in unite_to_6:
            intermediate_mapping[old_class] = 6
        elif old_class in unite_to_15:
            intermediate_mapping[old_class] = 15
        else:
            intermediate_mapping[old_class] = old_class
    
    # Passo 2: Raccogli tutte le classi uniche che rimangono (in ordine)
    unique_classes = sorted(set(v for v in intermediate_mapping.values() if v is not None))
    
    # Passo 3: Crea la mappa finale: classe originale -> numero della nuova classe
    final_mapping = {}
    for old_class in range(21):
        if intermediate_mapping[old_class] is None:
            final_mapping[old_class] = None
        else:
            intermediate_class = intermediate_mapping[old_class]
            new_class_id = unique_classes.index(intermediate_class)
            final_mapping[old_class] = new_class_id
    
    return final_mapping


def should_remove_annotation(class_id):
    """Verifica se un'annotazione deve essere rimossa (classe 0 o 12)"""
    return class_id in {0, 12}


def remap_class(old_class, mapping):
    """Rimappa una classe secondo la mappa fornita"""
    return mapping.get(old_class)


def print_class_mapping_info(class_mapping):
    """Stampa la mappa delle classi originali con i nuovi numeri assegnati"""
    # Nomi delle classi originali
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
    
    print("\n" + "="*90)
    print("MAPPA DELLE CLASSI (Originale -> Nuova)")
    print("="*90)
    print(f"{'Classe Orig.':<15} {'Nome Originale':<35} {'Classe Nuova':<15} {'Azione':<20}")
    print("-"*90)
    
    for old_class in range(21):
        new_class = class_mapping.get(old_class)
        name = class_names.get(old_class, "Sconosciuto")
        
        if new_class is None:
            action = "RIMOSSA"
        elif new_class == old_class:
            action = "Mantenuta"
        else:
            action = "Rinumerata"
        
        # Mostra solo le classi che vengono mantenute o che erano presenti
        if new_class is not None:
            print(f"{old_class:<15} {name:<35} {new_class:<15} {action:<20}")
        else:
            print(f"{old_class:<15} {name:<35} {'---':<15} {action:<20}")
    
    print("="*90 + "\n")


def process_yolo_dataset(dataset_path):
    """
    Processa il dataset YOLO:
    1. Elimina label e immagini con classe 0 o 12
    2. Unisce classi 4, 8, 6 -> 6
    3. Unisce classi 15, 19 -> 15
    4. Rinomina tutte le altre classi
    """
    dataset_path = Path(dataset_path)
    
    # Crea la mappa delle classi
    class_mapping = create_class_mapping()
    print_class_mapping_info(class_mapping)
    
    # Cartelle da processare
    splits = ['train', 'val', 'test']
    stats = {'removed_files': 0, 'modified_files': 0, 'removed_annotations': 0}
    
    for split in splits:
        labels_dir = dataset_path / 'labels' / split
        images_dir = dataset_path / 'images' / split
        
        if not labels_dir.exists():
            print(f"Cartella {labels_dir} non trovata, saltando...")
            continue
        
        print(f"\nProcessando {split}...")
        
        # Processa tutti i file di label
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Found {len(label_files)} label files")
        
        for label_file in label_files:
            image_file = None
            
            # Trova l'immagine corrispondente
            base_name = label_file.stem
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = images_dir / f"{base_name}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            # Leggi il file di label
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Errore nel leggere {label_file}: {e}")
                continue
            
            # Processa le annotazioni
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                try:
                    flag = parts[0]  # Prima colonna: flag
                    class_id = int(parts[1])  # Seconda colonna: classe
                except ValueError:
                    print(f"Errore nel parsing della classe in {label_file}: {parts[1] if len(parts) > 1 else 'N/A'}")
                    continue
                
                # Verifica se la classe deve essere eliminata
                if should_remove_annotation(class_id):
                    stats['removed_annotations'] += 1
                    continue
                
                # Rimappa la classe
                new_class = remap_class(class_id, class_mapping)
                if new_class is None:
                    stats['removed_annotations'] += 1
                    continue
                
                # Ricostruisci la riga con flag e nuova classe
                new_line = f"{flag} {new_class} " + " ".join(parts[2:]) + "\n"
                new_lines.append(new_line)
            
            # Se non ci sono più annotazioni, elimina file label e immagine
            if not new_lines:
                label_file.unlink()
                if image_file and image_file.exists():
                    image_file.unlink()
                stats['removed_files'] += 1
                print(f"  Eliminato: {label_file.name} (e immagine corrispondente)")
            else:
                # Scrivi il file di label modificato
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
                stats['modified_files'] += 1
                print(f"  Modificato: {label_file.name}")
    
    # Stampa le statistiche
    print("\n" + "="*50)
    print("STATISTICHE FINALE:")
    print(f"File rimossi (con tutte le annotazioni rimosse): {stats['removed_files']}")
    print(f"File modificati: {stats['modified_files']}")
    print(f"Annotazioni rimosse: {stats['removed_annotations']}")
    print("="*50)
    
    # Salva anche la mappa in un file di testo
    mapping_file = dataset_path.parent / 'class_mapping.txt'
    with open(mapping_file, 'w') as f:
        f.write("MAPPA DELLE CLASSI (Originale -> Nuova)\n")
        f.write("="*90 + "\n")
        f.write(f"{'Classe Orig.':<15} {'Nome Originale':<35} {'Classe Nuova':<15} {'Azione':<20}\n")
        f.write("-"*90 + "\n")
        
        class_names = {
            0: "unknown_tool", 1: "dissector", 2: "scissors", 3: "suction",
            4: "grasper 3", 5: "harmonic", 6: "grasper", 7: "bipolar",
            8: "grasper 2", 9: "cautery (hook, spatula)", 10: "ligasure", 11: "stapler",
            12: "unknown_tti", 13: "coagulation", 14: "other", 15: "retract and grab",
            16: "blunt dissection", 17: "energy - sharp dissection", 18: "staple",
            19: "retract and push", 20: "cut - sharp dissection"
        }
        
        for old_class in range(21):
            new_class = class_mapping.get(old_class)
            name = class_names.get(old_class, "Sconosciuto")
            
            if new_class is None:
                action = "RIMOSSA"
            elif new_class == old_class:
                action = "Mantenuta"
            else:
                action = "Rinumerata"
            
            if new_class is not None:
                f.write(f"{old_class:<15} {name:<35} {new_class:<15} {action:<20}\n")
            else:
                f.write(f"{old_class:<15} {name:<35} {'---':<15} {action:<20}\n")
    
    print(f"\nMappa delle classi salvata in: {mapping_file}")


if __name__ == '__main__':
    # Specifica il percorso del dataset
    dataset_path = './dataset'
    
    
    
    print(f"Processando dataset in: {dataset_path}")
    process_yolo_dataset(dataset_path)
    print("\nProcessamento completato!")
