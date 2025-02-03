import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn
from urllib.parse import unquote

class LogitLens:
    def __init__(self, layers, head, tokenizer=None, processor=None, output_attentions=False):
        """
        Универсальный Logit Lens для анализа скрытых представлений.
        :param layers: Список слоёв модели (например, transformer encoder layers)
        :param head: Выходная голова для получения логитов
        :param tokenizer: Токенизатор для текстового анализа
        :param processor: Процессор изображений
        """
        self.layers = layers
        self.head = head
        self.tokenizer = tokenizer
        self.processor = processor
        self.activations = {}

    def _register_hooks(self, modules: nn.Module | nn.ModuleList):
        """ Устанавливает forward-хуки для сохранения активаций слоёв. """
        def hook_fn(module, inputs, outputs):
            layer_idx = len(self.activations)
            self.activations[layer_idx] = outputs[0].detach()
        
        if isinstance(modules, (list, nn.ModuleList)):
            for module in modules:
                module.register_forward_hook(hook_fn)
        else:
            modules.register_forward_hook(hook_fn)

    def cleanup(self):
        """ Обнуляет прошлые активации. 
        """
        self.activations = {}

    def register(self):
        """ Подключает хуки и обнуляет прошлые активации. 
        """
        self.activations = {}
        
        self._register_hooks(self.layers)

    
    def visualize_text_predictions(self, norm=lambda x: x, top_k=5):
        """ Визуализирует топ-K токенов на разных слоях в виде тепловой карты. """
        assert self.tokenizer, "Не указан токенизатор!"

        logits_per_layer = {layer: self.head(norm(hidden)) for layer, hidden in self.activations.items()}
        layers = list(logits_per_layer.keys())
        num_layers = len(layers)
        
        probs_matrix = np.zeros((num_layers, top_k))
        tokens_matrix = np.empty((num_layers, top_k), dtype=object)
        
        for i, layer in enumerate(layers):
            logits = logits_per_layer[layer]
            if logits.ndim == 1:
                logits = logits[None, None, :]
            if logits.ndim == 2:
                logits = torch.unsqueeze(logits, 0)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            top_tokens = [self.tokenizer.convert_ids_to_tokens(tid.item()) for tid in top_ids[0]]
            probs_matrix[i, :] = top_probs.cpu().detach().float().numpy()
            tokens_matrix[i, :] = [unquote(tok) for tok in top_tokens]
        
        plt.figure(figsize=(10, num_layers * 0.5))
        ax = sns.heatmap(probs_matrix, annot=tokens_matrix, cmap="coolwarm", fmt="", xticklabels=False)
        ax.set_yticklabels([f"Layer {layer + 1}" for layer in layers], rotation=0)
        plt.title("Top-K Token Probabilities Across Layers (last token)")
        plt.xlabel("Top-K Tokens")
        plt.ylabel("Layers")
        plt.show()
        print(tokens_matrix)

    def print_top1_per_layer(self, norm=lambda x: x, mask=None):
        """Выводит top-1 токен на каждом слое, скрывая ненужные токены по маске."""
        assert self.tokenizer, "Не указан токенизатор!"
        
        logits_per_layer = {layer: self.head(norm(hidden)) for layer, hidden in self.activations.items()}
        layers = list(logits_per_layer.keys())
        num_layers = len(layers)
        max_tokens = max(logits.shape[1] for logits in logits_per_layer.values())
        
        probs_matrix = np.zeros((num_layers, max_tokens))
        tokens_matrix = np.empty((num_layers, max_tokens), dtype=object)
        
        for i, layer in enumerate(layers):
            logits = logits_per_layer[layer]
            if logits.ndim == 1:
                logits = logits[None, None, :]
            if logits.ndim == 2:
                logits = torch.unsqueeze(logits, 0)
            
            probs = F.softmax(logits, dim=-1)
            top_ids = probs.argmax(dim=-1)  # Получаем индексы top-1 токена
            
            for token_idx in range(top_ids.shape[1]):
                token_id = top_ids[0, token_idx].item()
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                
                if mask is None or mask[token_idx]:
                    probs_matrix[i, token_idx] = probs[0, token_idx, token_id].cpu().detach().float().numpy()
                    tokens_matrix[i, token_idx] = token
                else:
                    probs_matrix[i, token_idx] = 0
                    tokens_matrix[i, token_idx] = ""

        probs_print = list()
        tokens_print = list()

        for i in range(len(probs_matrix)):
            order = probs_matrix[i].argsort()[::-1]
            probs_print.append(probs_matrix[i][order[:10]])
            tokens_print.append(tokens_matrix[i][order[:10]])

        plt.figure(figsize=(20, num_layers * 0.5))
        ax = sns.heatmap(np.array(probs_print), annot=np.array(tokens_print), cmap="coolwarm", fmt="", xticklabels=False)
        ax.set_yticklabels([f"Layer {layer + 1}" for layer in layers], rotation=0)
        plt.title("Top-1 Token Per Layer")
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.show()

    
    def visualize_vision_features(self, num_channels=8):
        """ Визуализирует 2D-карты активаций в vision encoder. """
        for layer, activation in self.activations.items():
            activation = activation.cpu().detach().float().numpy()
            if activation.ndim == 4:
                activation = activation[0]
                num_channels = min(num_channels, activation.shape[0])
                fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 2, 2))
                fig.suptitle(f"Vision Layer {layer + 1}")
                for i in range(num_channels):
                    ax = axes[i] if num_channels > 1 else axes
                    sns.heatmap(activation[i], cmap="coolwarm", ax=ax, cbar=False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Ch {i}")
                plt.show()
    
    def visualize_pca(self):
        """ PCA-анализ скрытых представлений """
        all_features = torch.cat([h.view(h.size(0), -1) for h in self.activations.values()], dim=0)
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_features.float().cpu().numpy())
        plt.figure(figsize=(6, 6))
        plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
        plt.title("PCA Projection of Hidden States")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
    
    def cosine_distance_heatmap(self, norm=None):
        """ Визуализирует матрицу косинусных расстояний между слоями. """
        layer_representations = [norm(h).view(h.size(0), -1).mean(dim=0) for h in self.activations.values()]
        similarities = torch.stack([F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
                                    for x in layer_representations for y in layer_representations])
        similarity_matrix = similarities.view(len(layer_representations), -1).detach().float().cpu().numpy()
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, cmap="coolwarm", annot=True)
        plt.title("Cosine Distance Between Layers")
        plt.xlabel("Layer")
        plt.ylabel("Layer")
        plt.show()

    def uncertainity(self, norm=None):
        """ Визуализирует матрицу косинусных расстояний между слоями. """
        layer_representations = [norm(h).view(h.size(0), -1).mean(dim=0) for h in self.activations.values()]
        similarities = torch.stack([F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
                                    for x in layer_representations for y in layer_representations])
        similarity_matrix = similarities.view(len(layer_representations), -1).float().cpu().numpy()
        return similarity_matrix.mean()

    def show_patches(self):
        pass


class StatiscitalLens:
    def __init__(self, activations, norm, head, tokenizer=None, processor=None):
        """
        Универсальный Logit Lens для анализа скрытых представлений.
        :param layers: Список слоёв модели (например, transformer encoder layers)
        :param head: Выходная голова для получения логитов
        :param tokenizer: Токенизатор для текстового анализа
        :param processor: Процессор изображений
        """

        self.activations = activations
        self.norm = norm
        self.head = head
        self.tokenizer = tokenizer
        self.processor = processor
        self.activations = {}