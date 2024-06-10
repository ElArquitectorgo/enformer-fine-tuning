time for i in `cat ../rna_data_filtered.txt|awk '{print $2}'|tr -d "+"|tr -d "-" |grep -v GTEX|grep -v CNhs|sort|uniq`; do
echo $i

if grep -q "$i" out.log; then
    echo "$i ya ha sido descargado."
	echo $i ya descargado >> out.log
	
elif wget https://www.encodeproject.org/files/${i}/@@download/${i}.bigWig; then
	echo $i bien >> out.log
else
	echo $i mal >> out.log
fi

done