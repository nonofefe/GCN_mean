declare -a array=()
declare -a array=("uniform")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_link_pred.py --rate $b --type ${array[i]} --dataset citeseer
        a=`expr $a + 1`
    done
}