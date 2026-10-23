# Questionário de Avaliação Técnica (50 Questões - V ou F)

Instruções:

- Preencha dentro dos colchetes com **V** para Verdadeiro ou **F** para Falso (ou **X**).
- Exemplo: `[V]` ou `[F]`.
- Quando terminar de preencher e salvar o arquivo, me avise para eu fazer a correção detalhada.

---

### Bloco 1: Imagens, Cores e NumPy

- [X] 01. No PyTorch, as imagens para convolução devem estar no formato (Canais, Altura, Largura), enquanto o NumPy e o PIL tipicamente carregam em (Altura, Largura, Canais).
- [X] 02. A divisão dos pixels por 255 é feita para converter os valores inteiros de 0 a 255 em números de ponto flutuante no intervalo entre 0,0 e 1,0.
- [ ] 03. Se invertermos a ordem dos canais de RGB para BGR, as convoluções aprendem da mesma forma sem qualquer impacto, pois a rede neural é indiferente à ordem física dos canais de cor.
- [ ] 04. O método ImageOps.fit realiza um corte central proporcional (Center Crop) para ajustar a foto ao tamanho desejado sem distorcer as proporções dos objetos.
- [ ] 05. Um lote de imagens no PyTorch possui 4 dimensões no formato: (Batch_size, Canais, Altura, Largura).

---

### Bloco 2: Ingestão de Dados e Hardware

- [X] 06. O mecanismo de Lazy Loading (leitura preguiçosa sob demanda) no Dataset serve para evitar carregar gigabytes de fotos de uma vez na memória RAM do computador.
- [ ] 07. O parâmetro shuffle=True deve ser utilizado obrigatoriamente tanto no DataLoader de treino quanto nos de validação e teste para garantir justiça nas métricas.
- [X] 08. O parâmetro num_workers cria processos paralelos na CPU para pré-carregar os próximos lotes de imagens enquanto a GPU processa o lote atual.
- [ ] 09. A memória RAM paginada convencional do sistema operacional permite transferências diretas para a VRAM da GPU via DMA sem necessidade de travamento de páginas (page-locking).
- [X] 10. Ativar pin_memory=True no DataLoader ajuda a acelerar a transferência de dados da RAM para a VRAM em placas de vídeo dedicadas via barramento PCIe.

---

### Bloco 3: Aumento de Dados / Data Augmentation

- [ ] 11. Filtros convolucionais padrão são naturalmente invariantes à rotação e ao espelhamento da imagem.
- [X] 12. O aumento de dados é aplicado no conjunto de treino para expandir artificialmente o espaço amostral e forçar a rede a reconhecer padrões em orientações variadas.
- [ ] 13. É uma boa prática de engenharia aplicar rotações e espelhamentos aleatórios também no conjunto de teste final para melhorar a nota de avaliação da rede.
- [X] 14. Ao rotacionar uma foto, ângulos excessivamente grandes geram cantos pretos que podem cobrir partes úteis da folha caso não sejam compensados.
- [X] 15. Realizar a rotação e o espelhamento antes do corte proporcional (ImageOps.fit) ajuda a eliminar parte das bordas pretas criadas pela rotação.

---

### Bloco 4: Convolução e Geometria

- [X] 16. Um filtro convolucional de tamanho 3x3 aplicado sobre uma imagem de entrada colorida (3 canais) possui exatamente 27 pesos (mais 1 viés).
- [X] 17. Numa convolução 2D, os mesmos 27 pesos do filtro são reutilizados em todas as posições espaciais da imagem conforme o filtro desliza (compartilhamento de pesos).
- [X] 18. Em uma camada com entrada de 3 canais e saída de 32 canais, existem 32 filtros convolucionais distintos varrendo a imagem.
- [ ] 19. Com kernel 3x3 e passo 1 (stride=1), se não utilizarmos padding (padding=0), a saída terá exatamente a mesma altura e largura que a imagem de entrada.
- [X] 20. O padding com valor 1 (padding=1) adiciona uma moldura de pixels nas bordas para compensar a perda de 2 pixels gerada pelo kernel 3x3.

---

### Bloco 5: Funções de Ativação

- [ ] 21. A intensidade de ativação dos neurônios intermediários em redes modernas com ReLU varia obrigatoriamente e estritamente no intervalo de 0,0 a 1,0.
- [X] 22. A função ReLU substitui qualquer valor negativo por 0 e entrega qualquer valor positivo idêntico ao original.
- [X] 23. Se uma rede neural profunda for composta apenas por camadas convolucionais e lineares sem nenhuma função de ativação não-linear, toda a rede se comporta como uma única função linear.
- [X] 24. O parâmetro inplace=True na ReLU modifica os dados diretamente no mesmo espaço de memória, economizando alocação de memória na GPU.
- [X] 25. A função Sigmoid foi substituída pela ReLU nas camadas internas de visão computacional porque saturava em valores altos/baixos, provocando o desaparecimento do gradiente (Vanishing Gradient).

---

### Bloco 6: Pooling e Redução Dimensional

- [X] 26. A camada nn.MaxPool2d(kernel_size=2, stride=2) corta a altura e a largura da imagem pela metade escolhendo o valor máximo de cada janela 2x2.
- [ ] 27. O MaxPool2d cria novos valores numéricos calculando a média ponderada dos pixels da janela.
- [X] 28. Preservar o valor máximo no MaxPool faz sentido porque números mais altos representam detecções mais fortes do padrão do filtro naquela região.
- [X] 29. A camada nn.AdaptiveAvgPool2d((1, 1)) calcula a média de todos os pixels restantes de cada canal, colapsando a dimensão espacial em 1x1 independente da resolução que entra.
- [X] 30. Usar AdaptiveAvgPool2d((1, 1)) antes da camada linear ajuda a evitar a explosão do número de parâmetros e reduz o risco de overfitting.

---

### Bloco 7: Normalização em Lote - BatchNorm

- [ ] 31. O nn.BatchNorm2d normaliza as ativações de cada canal para que tenham média próxima de 0 e desvio padrão próximo de 1 ao longo do lote.
- [ ] 32. O BatchNorm é dispensável se você já tiver normalizado a imagem de entrada entre 0 e 1 no NumPy, pois as multiplicações das convoluções mantêm a normalização intacta.
- [X] 33. Na matemática da normalização, se todas as entradas de um canal forem multiplicadas por um fator k, esse fator k se cancela na fração.
- [X] 34. O cancelamento de escala do BatchNorm ajuda a suavizar o terreno da perda (loss landscape), evitando penhascos íngremes e variações descontroladas do gradiente.
- [ ] 35. Durante a avaliação ou inferência (model.eval()), o BatchNorm continua calculando a média e o desvio padrão do lote atual em vez de usar os valores históricos acumulados no treino.

---

### Bloco 8: Regularização e Camadas Finais

- [X] 36. A camada nn.Dropout(p=0.4) desliga aleatoriamente 40% dos neurônios durante o treino para evitar que eles fiquem dependentes uns dos outros (co-adaptação).
- [ ] 37. Na fase de teste ou produção (model.eval()), o Dropout continua desligando 40% dos neurônios aleatoriamente.
- [ ] 38. A camada nn.Flatten() altera os valores numéricos dos pixels para prepará-los para a camada linear.
- [X] 39. O nn.Flatten() converte o tensor 4D de formato (Batch, Canais, 1, 1) em uma matriz 2D de formato (Batch, Features) sem alterar nenhum valor numérico.
- [X] 40. Diferente da convolução, que olha para janelas locais 3x3, a camada nn.Linear conecta cada um de seus neurônios a todas as entradas recebidas da camada anterior.

---

### Bloco 9: Saídas, Erro e Função de Perda

- [X] 41. As pontuações brutas de saída geradas pela última camada linear antes de qualquer ativação são chamadas de logits.
- [ ] 42. Os logits só podem assumir valores positivos entre 0 e 100.
- [X] 43. A função Softmax transforma os logits em probabilidades percentuais cuja soma é obrigatoriamente 100% (ou 1,0).
- [ ] 44. A nn.CrossEntropyLoss do PyTorch exige que você aplique a função Softmax manualmente no modelo antes de passar os dados para ela.
- [X] 45. Se a rede atribuir uma probabilidade quase nula à classe que era o gabarito correto da foto (por exemplo, 0,0001%), a função CrossEntropyLoss gerará um valor de erro muito alto.

---

### Bloco 10: Otimização e Ciclo de Treino

- [X] 46. A chamada optimizer.zero_grad() é necessária no início de cada passo para evitar que os gradientes dos lotes anteriores se acumulem indefinidamente.
- [ ] 47. A linha loss.backward() é responsável por alterar diretamente os valores dos pesos e vieses da rede neural.
- [X] 48. O método optimizer.step() é quem realmente lê os gradientes calculados e atualiza os pesos na direção que diminui o erro.
- [X] 49. O comando with torch.no_grad(): desativa o cálculo e armazenamento de derivadas, economizando memória VRAM e acelerando a avaliação.
- [X] 50. Salvar o modelo com torch.save(model.state_dict(), "modelo.pth") armazena apenas o dicionário com os valores numéricos dos pesos e vieses, sendo a forma recomendada em PyTorch.
