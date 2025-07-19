#!\bin\bash 
mkdir -p ../data/bindingdb
cd ../data/bindingdb
wget https://bindingdb.org/rwd/bind/downloads/BindingDB_All_202507_tsv.zip
unzip BindingDB_All_202507_tsv.zip -d .
mv BindingDB_All_202507_tsv/* .
rm -r BindingDB_All_202507_tsv
cd -
echo "BindingDB data downloaded and extracted to ../data/bindingdb"
echo "Script completed successfully."
exit 0