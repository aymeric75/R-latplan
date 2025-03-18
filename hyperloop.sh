#!/bin/bash


exp_dir="/workspace/R-latplan/r_latplan_exps/hanoi/hanoi_complete_clean_faultless_withoutTI"
sub_exp_dir=confsBISBIS
exps_dir=$exp_dir/$sub_exp_dir


### construction des hyper params



### ajouter l'algo de search: blind, lama, lmcut, mands

### si prend trop de temps annuler

### calculer le nbre total de génération : 192

### 1) "training"
###
###     in "hanoi_complete_clean_faultless_withoutTI" create
###         a "confs" dir (for all) and within for each 
###           a sub conf named: 
###             filter_out_dt1__factorize_dt1__filter_out_dt2__only_supersets_dt2__x_all_atoms_dt2__x_atoms_from_cond_effects_dt2__x_atoms_outisde_of_cond_effects_dt2__heury
###
###     doter decision_tree_multisklearn_NO_SHAPS.py de params
### _
###     
###     renommer et placer le PDDL
###           decision_tree_multi_sklearn_NO_SHAPS produit un pddl
###
###
####       ensuite



# create exp dir
if [ ! -d "$exps_dir" ]; then
  mkdir $exps_dir
fi




# 2) testing (go over each dir)

# counter=0
# ####### FIRST LOOP: CRE TING THE EXP DIRS AND GENERATING THE NEW PDDL


# for filter_out_dt1 in true false; do
#   for factorize_dt1 in true false; do
#     for filter_out_dt2 in true false; do

#       # for filter_out_dt1 in true; do
#       #   for factorize_dt1 in true; do
#       #     for filter_out_dt2 in true; do
#       #for only_supersets_dt2 in true false; do
#       #echo "filter_out_dt1=$filter_out_dt1, factorize_dt1=$factorize_dt1, filter_out_dt2=$filter_out_dt2, only_supersets_dt2=$only_supersets_dt2"
#       # Add your logic here, using the four variables

#       for var_x in x_all_atoms_dt2 x_atoms_from_cond_effects_dt2 x_atoms_outisde_of_cond_effects_dt2; do
#         x_all_atoms_dt2=false
#         x_atoms_from_cond_effects_dt2=false
#         x_atoms_outisde_of_cond_effects_dt2=false

#         #declare $var_x=true

#         on_x=--$var_x
#         # echo $on_x

#         echo "doing pddl $counter / 23"


#         dir_name="${counter}__${filter_out_dt1}__${factorize_dt1}__${filter_out_dt2}__${var_x}"

#         if [ ! -d "$exps_dir/$dir_name" ]; then
#           mkdir $exps_dir/$dir_name
#         fi


#         echo "in $dir_name"

#         python $exp_dir/decision_tree_multi_sklearn_NO_SHAPS__.py --filter_out_dt1 $filter_out_dt1 --factorize_dt1 $factorize_dt1 --filter_out_dt2 $filter_out_dt2 $on_x --index $counter

#         # #exit 1


#         #### une fois le putain de bordel fait, le PDDL ne change pas, c'est juste la SEARCH qui change

#         # echo "x_all_atoms_dt2=$x_all_atoms_dt2, x_atoms_from_cond_effects_dt2=$x_atoms_from_cond_effects_dt2, x_atoms_outisde_of_cond_effects_dt2=$x_atoms_outisde_of_cond_effects_dt2"
#         # Add your logic here

#         #for heuri in blind lama lmcut mands; do
#         for heuri in blind lama lmcut; do
#           #echo "Current value: $value"
#           # Add your logic here
        

#           cp -r $exp_dir/pbs_virgin $exps_dir/$dir_name/

#           rm -rdf $exps_dir/$dir_name/pbs_$heuri

#           mv $exps_dir/$dir_name/pbs_virgin $exps_dir/$dir_name/pbs_$heuri
#           # echo $exps_dir/$dir_name/pbs


          


#         done

#         ((counter++))

#       done

#     done
#   done
# done


# # echo $counter

# exit 1

#####  SECOND LOOP : GO OVER THE EXPO DIRS AND GENERATE THE PLANS


counter=0
####### FIRST LOOP: CRE TING THE EXP DIRS AND GENERATING THE NEW PDDL

# 15__false__true__false__x_all_atoms_dt2

# for filter_out_dt1 in true false; do
#   for factorize_dt1 in true false; do
#     for filter_out_dt2 in true false; do


for filter_out_dt1 in true false; do
  for factorize_dt1 in true false; do
    for filter_out_dt2 in true false; do

      #for only_supersets_dt2 in true false; do
      #echo "filter_out_dt1=$filter_out_dt1, factorize_dt1=$factorize_dt1, filter_out_dt2=$filter_out_dt2, only_supersets_dt2=$only_supersets_dt2"
      # Add your logic here, using the four variables

      for var_x in x_all_atoms_dt2 x_atoms_from_cond_effects_dt2 x_atoms_outisde_of_cond_effects_dt2; do
        x_all_atoms_dt2=false
        x_atoms_from_cond_effects_dt2=false
        x_atoms_outisde_of_cond_effects_dt2=false

        #declare $var_x=true

        on_x=--$var_x
        # echo $on_x

        dir_name="${counter}__${filter_out_dt1}__${factorize_dt1}__${filter_out_dt2}__${var_x}"

        if [ ! -d "$exps_dir/$dir_name" ]; then
          mkdir $exps_dir/$dir_name
        fi

        #for heuri in blind lama lmcut mands; do
        #echo "Current value: $value"
        # Add your logic here
      
        # cp -r $exp_dir/pbs_virgin $exps_dir/$dir_name/

        # rm -rdf $exps_dir/$dir_name/pbs_$heuri

        # mv $exps_dir/$dir_name/pbs_virgin $exps_dir/$dir_name/pbs_$heuri
        # # echo $exps_dir/$dir_name/pbs
        python r_latplan_testing.py r_latplan gen_plans hanoi hanoi_complete_clean_faultless_withoutTI/$sub_exp_dir/$dir_name --use_base_to_load yes
        
        # if [ "$counter" -ge 15 ]; then
        #   python r_latplan_testing.py r_latplan gen_plans hanoi hanoi_complete_clean_faultless_withoutTI/confs/$dir_name --use_base_to_load yes
        # fi
        #done

        ((counter++))

      done
    done
  done
done







######  THIRD LOOP : GO OVER THE EXPO DIRS AND PBS AND GENERATE THE STATS