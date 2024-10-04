import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, n_categoricas, n_numericas, n_labels, embedding_size=10, hidden_units=[100, 100, 100], drop_rate=0.3):
        super(DeepFM, self).__init__()

        # Parte de embeddings para las variables categóricas
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=embedding_size, padding_idx=0)
            for n_cat in n_categoricas
        ])

        # Parte DNN: Capa densa que incluye embeddings y variables numéricas
        input_size = len(n_categoricas) * embedding_size + n_numericas
        self.dnn = nn.Sequential()
        for i, units in enumerate(hidden_units):
            self.dnn.add_module(f'dense_{i}', nn.Linear(input_size, units))
            self.dnn.add_module(f'relu_{i}', nn.ReLU())
            self.dnn.add_module(f'dropout_{i}', nn.Dropout(drop_rate))
            self.dnn.add_module(f'batchnorm_{i}', nn.BatchNorm1d(units))
            input_size = units

        # Salidas para cada etiqueta binaria (sigmoid)
        self.output_layers = nn.ModuleList([nn.Linear(hidden_units[-1], 1) for _ in range(n_labels)])

        # Capa lineal para ajustar las dimensiones de FM
        self.fm_linear = nn.Linear(embedding_size, hidden_units[-1])  # embedding_size=15, hidden_units[-1]=100

    def forward(self, x_categorico, x_numerico):
        # Embedding de las variables categóricas
        emb_x = [emb(x_categorico[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_x = torch.stack(emb_x, dim=1)  # [batch_size, num_fields, embedding_size]
        
        # Factorization Machine (FM) - Segundo Orden
        square_of_sum = torch.sum(emb_x, dim=1) ** 2  # [batch_size, embedding_size]
        sum_of_square = torch.sum(emb_x ** 2, dim=1)  # [batch_size, embedding_size]
        fm_second_order = 0.5 * (square_of_sum - sum_of_square)  # [batch_size, embedding_size]
        
        #print(f'fm_second_order shape before fm_linear: {fm_second_order.shape}')  # [128,15]
        
        # Ajustar el tamaño de fm_second_order para que coincida con dnn_output
        fm_second_order_adjusted = self.fm_linear(fm_second_order)  # [128,100]
        #print(f'fm_second_order_adjusted shape: {fm_second_order_adjusted.shape}')  # [128,100]
        
        # Aplanar embeddings
        emb_x_flat = emb_x.view(emb_x.size(0), -1)  # [128,75]
        #print(f'emb_x_flat shape: {emb_x_flat.shape}')  # [128,75]
        
        # Asegurarse de que x_numerico tenga la forma adecuada
        if x_numerico.dim() == 1:
            x_numerico = x_numerico.unsqueeze(1)
        #print(f'x_numerico shape: {x_numerico.shape}')  # [128,45] si n_numericas=45
        
        # Concatenar emb_x_flat y x_numerico
        x_combined = torch.cat([emb_x_flat, x_numerico], dim=1)  # [128,120]
        #print(f'x_combined shape: {x_combined.shape}')  # [128,120] si num_fields=5, embedding_size=15, n_numericas=45
        
        # Parte DNN
        dnn_output = self.dnn(x_combined)  # [128,100]
        #print(f'dnn_output shape: {dnn_output.shape}')  # [128,100]
        
        # Sumar las salidas de la DNN y FM
        combined_output = dnn_output + fm_second_order_adjusted  # [128,100]
        #print(f'combined_output shape: {combined_output.shape}')  # [128,100]
        
        # Salidas finales
        outputs = [torch.sigmoid(output_layer(combined_output)) for output_layer in self.output_layers]
        
        # Convertir la lista de salidas a un tensor
        return torch.cat(outputs, dim=1)