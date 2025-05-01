#!/bin/bash

# Définition du dossier de base (un niveau plus haut)
BASE_DIR="$(dirname "$(pwd)")/confsBISBISBIS"

# echo $BASE_DIR
# exit 1

# Vérification que le dossier existe
if [ ! -d "$BASE_DIR" ]; then
    echo "Le dossier $BASE_DIR n'existe pas."
    exit 1
fi

# Fichier de sortie
OUTPUT_FILE="$BASE_DIR/results.txt"
echo "config mean_ratio total_plans" > "$OUTPUT_FILE"

# Parcourir tous les sous-dossiers
for superfold in "$BASE_DIR"/*/; do
    superfold_name=$(basename "$superfold")
    
    #echo $superfold_name

    # Chercher tous les sous-dossiers nommés pbs_*
    for pbs_truc in "$superfold"pbs_*/; do
        [ -d "$pbs_truc" ] || continue
        config="${superfold_name}_$(basename "$pbs_truc")"
        
        # Initialisation des compteurs
        total_plan_files=0
        total_ratio=0
        valid_ratios=0
        
        echo "Analyse du dossier $pbs_truc..."
        
        # Parcourir récursivement les fichiers .plan
        while IFS= read -r -d '' plan_file; do
            ((total_plan_files++))
            
            # Extraire le dossier parent du fichier
            parent_dir=$(basename $(dirname "$plan_file"))
            
            # Extraire la première partie du nom du dossier (avant "_")
            groundtruth=$(echo "$parent_dir" | cut -d'_' -f1)
            
            # Vérifier que groundtruth est bien un entier
            if ! [[ "$groundtruth" =~ ^[0-9]+$ ]]; then
                echo "Fichier: $plan_file - Erreur: Groundtruth non valide ($groundtruth)"
                continue
            fi
            
            # Compter le nombre de lignes avant "cost"
            line_count=$(grep -n "cost" "$plan_file" | cut -d: -f1 | head -n 1)
            
            if [ -z "$line_count" ]; then
                echo "Fichier: $plan_file - Mot 'cost' non trouvé - Groundtruth: $groundtruth"
            else
                ratio=$(echo "scale=4; $groundtruth / $line_count" | bc)
                total_ratio=$(echo "scale=4; $total_ratio + $ratio" | bc)
                ((valid_ratios++))
            fi
        done < <(find "$pbs_truc" -type f -name "*.plan" -print0)
        
        # Calculer la moyenne des ratios si des valeurs valides existent
        if [ "$valid_ratios" -gt 0 ]; then
            average_ratio=$(echo "scale=4; $total_ratio / $valid_ratios" | bc)
        else
            average_ratio="N/A"
        fi
        
        # Écrire les résultats dans le fichier
        echo "$config $average_ratio $total_plan_files" >> "$OUTPUT_FILE"
    done

done

echo "Résultats sauvegardés dans $OUTPUT_FILE"