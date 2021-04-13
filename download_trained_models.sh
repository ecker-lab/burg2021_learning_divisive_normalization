#!/bin/bash
cd cnn3
wget -O cnn3_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/BF03XE
tar -xf cnn3_weights.tar.gz
cd ../divisive_3x3_surround_net
wget -O divisive_3x3_surround_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/LUOVY7
tar -xf divisive_3x3_surround_net_weights.tar.gz
cd ../divisive_5x5_surround_net
wget -O divisive_5x5_surround_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/WVGXY9
tar -xf divisive_5x5_surround_net_weights.tar.gz
cd ../divisive_7x7_surround_net
wget -O divisive_7x7_surround_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/BQFSJ2
tar -xf divisive_7x7_surround_net_weights.tar.gz
cd ../divisive_net
wget -O divisive_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/ZONTHV
tar -xf divisive_net_weights.tar.gz
cd ../nonspecific_divisive_net
wget -O nonspecific_divisive_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/LC9INP
tar -xf nonspecific_divisive_net_weights.tar.gz
cd ../subunit_net
wget -O subunit_net_weights.tar.gz https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/0JCXYO/S6QDOJ
tar -xf subunit_net_weights.tar.gz
