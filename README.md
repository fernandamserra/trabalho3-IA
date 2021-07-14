# Terceiro trabalho prático de IA
Arquivos usados para solução e desenvolvimento de relatório sobre o problema de classificação de trens apresentado por Michalski.

# Como executar
Primeiramente instale a versão mais recente do `python3` disponível no seu gerenciador de pacotes

###### ubuntu
```bash
sudo apt update
sudo apt install python3
```

Em seguida siga os passos abaixo para instalar a biblioteca `Logic Tensor Networks` 

###### ubuntu
```bash
git clone https://github.com/logictensornetworks/logictensornetworks.git
python3 -m pip install -e logictensornetworks
```

Por fim, para excutar o código entre na pasta clonada deste repositório e execute o comando abaixo
```bash
python3 LTN_Michalski_o_retorno.py
```

Se tudo funcionar como esperado serão exibidos alguns `warnings` e em seguida serão exibidas as epochs e será gerado um arquivo chamado `train_results.csv` com histórico completo de treinos

# Notas
* Os gráficos desse trabalho podem ser encontrados em nosso relatório junto com nossas explicações e observações do problema resolvida
