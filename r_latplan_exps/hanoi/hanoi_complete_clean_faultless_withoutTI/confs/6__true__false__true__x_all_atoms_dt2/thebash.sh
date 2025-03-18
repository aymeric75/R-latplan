#!/bin/bash

# Définition du dossier de base
BASE_DIR="pbs_lama"

# Vérification que le dossier existe
if [ ! -d "$BASE_DIR" ]; then
    echo "Le dossier $BASE_DIR n'existe pas."
    exit 1
fi

# Initialisation des compteurs
total_plan_files=0
total_ratio=0
valid_ratios=0

echo "Analyse des fichiers .plan dans $BASE_DIR et ses sous-dossiers..."

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
        echo "Fichier: $plan_file - Lignes avant 'cost': $line_count - Groundtruth: $groundtruth - Ratio: $ratio"
    fi

done < <(find "$BASE_DIR" -type f -name "*.plan" -print0)

# Calculer la moyenne des ratios si des valeurs valides existent
if [ "$valid_ratios" -gt 0 ]; then
    average_ratio=$(echo "scale=4; $total_ratio / $valid_ratios" | bc)
    echo "Moyenne des ratios: $average_ratio"
else
    echo "Aucun ratio valide calculé."
fi

echo "Total des fichiers .plan trouvés: $total_plan_files"