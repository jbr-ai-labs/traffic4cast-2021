#!/bin/bash

echo "Downloading into folder:" $1

# echo " Downloading Istanbul:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/ISTANBUL.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=ubh1Xe2HWgI1OcMq55HSbpcjcZ8%3D&Expires=1641218478&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfeqQMc_P-fU_-KyfWYOdaHH_X1EaIvyhBq1NNVqqPaICLkCrGVVuzp61bwzr88cektAWA0hu4StY3YxK1geOFwxsE754w9hlxchGxhtZSg" -o $1/ISTANBUL.tar.gz
#
# echo " Downloading Berlin:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/BERLIN.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=qO9yboHnm13%2B%2Bu7QQwKhAAVpMa8%3D&Expires=1641218478&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfTDJ9NilrYs80uKCj8r4QHirkg5RmIYKu5OkDRZYk8Cqq6BtWToRPWFMnrJaHMNUDL8EeCp5cPG9ehGgx-REWhds5Ez4Xd31bp1Og0dLBg" -o $1/BERLIN.tar.gz
#
#
# echo " Downloading Melbourne:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/MELBOURNE.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=H2BhxUhgilClr%2FYMOGdG36K7Hko%3D&Expires=1641218479&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfea6d7VeKkg5h3xdks9kpR-C_FJ-wFSW4xvCsj3WdQRSVD3Nj_XErA-waJ4Kt8rIf4fPeNZudBeCfh9bBdbEPCVikGTqLYBi2bYzu_0XKg" -o $1/MEBOURNE.tar.gz
#
# echo " Downloading Chicago:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/CHICAGO.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=xNThWVz1G9T42lqhYV5XEsI%2Fia4%3D&Expires=1641218479&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfbzOqv-d3QeJovgYHwHtNn9VDqNTU0NE9F6XDGgSRTDmvKuMpUR6Yj4cRjS2jkppzhGpqA0_Eba_yBF-K1IBgQoy4ZVUVF3ZgDFDC_66rg" -o $1/CHICAGO.tar.gz
#
# echo " Downloading Barcelona:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/BARCELONA.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=Pdi7x2n77AE5HxyxKkrx%2FhcsyJc%3D&Expires=1641218479&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfZfzEsUtdfq8cTWsbKrX0VUX_t9hXPnO-I5sAJVLLEk80qa6w-xDwB2jSQBf64p9UGRKDBk_OPCJHDkc_O3NwMW42GJ0XXdF0gJRrpjNzg" -o $1/BARCELONA.tar.gz
#
# echo " Downloading Bangkok:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/BANGKOK.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=g3X33RsepxmT7QhE3Xk7Q71vfds%3D&Expires=1641218480&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfeJrMYWWznqwmzyjMoSv0FQDukOraFP5zMI75L0B5IM43lKhNWW1lu_wEoUWMxT9cxpAjzMaZKvf3rb4YbbZWNAXxNgLsWQ4VF9g3AEaIg" -o $1/BANGKOK.tar.gz
#
# echo " Downloading Moscow:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/MOSCOW.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=wmeHFaeLFj1%2Fey%2FcBkyVcjq4SeM%3D&Expires=1641218480&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfVLY8sJOKGxUVb9rNDTuSuyH-JhXAy8_SRIXC_dCHzH-6DjNpXxdeChFE8z-ju_y_CIkTi8w-HTmu3CEOi_Zxr6b5UUHt6ll2Ndq_-ikdQ" -o $1/MOSCOW.tar.gz
#
# echo " Downloading Antwerp:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/ANTWERP.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=TMFl5ZQKSMANZn%2BrvB3bsA5vTKc%3D&Expires=1641218481&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfUHGOjIJt0gGz7t_8wtfev_I9fp-OVM9u6b7F2k3vfagr0F2zk6g3_ClZY1dougBG2kKmfsVbK_dhKD-QOBeZFHlMwRXvYA8q_jRPCiUtg" -o $1/ANTWERP.tar.gz
#
# echo " Downloading New York:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/NEWYORK.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=FoCxdttXav%2F5PspVxJdOeYmRRTY%3D&Expires=1641218481&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfVcazccx20fUSgMPM8zB6pHZZbqYddjohOSDTm4pPko2JaNzhSij7GSWnTmrW_asHqQmUdUJjlja8Y2LXfoUn3ytig8CWn7uVwrhbQaIpA" -o $1/NEWYORK.tar.gz
#
# echo " Downloading Vienna:"
# curl "https://lake-iarai-us-east-1.s3.amazonaws.com/trafficast2021/releases/2021-06-15/VIENNA.tar.gz?AWSAccessKeyId=AKIA2FGGSICU7UR27IGO&Signature=%2BP0FFNyLqIV%2FbiEH%2BFb2Bzez2l8%3D&Expires=1641218496&mkt_tok=MTQyLVVFTC0zNDcAAAF-BnFlfeQE9wSIqxovxMJ6PZO-OuuU06l8roN3gmgyMtMhli-yyIYPFRsw4EDJq66cbTAgz4ONrv9mGxScfLbC2PRZ04N95NE6CunOpMVhkCCzyA" -o $1/VIENNA.tar.gz
#


echo "Extracting:"

cd $1

ls

tar -xzvf ISTANBUL.tar.gz

tar -xzvf BERLIN.tar.gz

tar -xzvf MELBOURNE.tar.gz

tar -xzvf CHICAGO.tar.gz

tar -xzvf BARCELONA.tar.gz

tar -xzvf BANGKOK.tar.gz

tar -xzvf MOSCOW.tar.gz

tar -xzvf ANTWERP.tar.gz

tar -xzvf NEWYORK.tar.gz

tar -xzvf VIENNA.tar.gz

rm ISTANBUL.tar.gz

rm BERLIN.tar.gz

rm MELBOURNE.tar.gz

rm CHICAGO.tar.gz

rm BARCELONA.tar.gz

rm BANGKOK.tar.gz

rm MOSCOW.tar.gz

rm ANTWERP.tar.gz

rm NEWYORK.tar.gz

rm VIENNA.tar.gz

