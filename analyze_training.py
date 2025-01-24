import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_metrics(model_dir):
    """Ladda träningsmetriker från sparad fil"""
    metrics_path = os.path.join(model_dir, 'training_metrics.npy')
    return np.load(metrics_path, allow_pickle=True).item()

def plot_learning_curves(metrics, save_dir=None):
    """Plotta inlärningskurvor med detaljerad analys"""
    plt.figure(figsize=(12, 8))
    
    # Plotta förluster
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epochs'], metrics['train_losses'], 'b-', label='Träningsförlust')
    plt.plot(metrics['epochs'], metrics['val_losses'], 'r-', label='Valideringsförlust')
    plt.axvline(x=metrics['best_epoch'], color='g', linestyle='--', label='Bästa epoch')
    plt.title('Tränings- och Valideringsförlust')
    plt.xlabel('Epoch')
    plt.ylabel('Förlust')
    plt.legend()
    plt.grid(True)
    
    # Plotta förlustskilland
    plt.subplot(2, 1, 2)
    loss_diff = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    plt.plot(metrics['epochs'], loss_diff, 'g-', label='Val - Train förlust')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Överanpassningsanalys')
    plt.xlabel('Epoch')
    plt.ylabel('Förlustskillnad')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'training_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.show()

def analyze_training(metrics):
    """Analysera träningsresultat och ge rekommendationer"""
    train_losses = np.array(metrics['train_losses'])
    val_losses = np.array(metrics['val_losses'])
    loss_diff = val_losses - train_losses
    
    analysis = {
        'best_epoch': metrics['best_epoch'],
        'best_val_loss': metrics['best_val_loss'],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'overfitting_score': loss_diff[-1],
        'convergence_rate': (train_losses[0] - train_losses[-1]) / len(train_losses)
    }
    
    # Generera rekommendationer
    recommendations = []
    
    if analysis['overfitting_score'] > 0.1:
        recommendations.extend([
            "Tecken på överanpassning. Prova:",
            "- Öka weight decay (nu: 0.02)",
            "- Lägg till mer dropout",
            "- Minska modellens storlek",
            "- Samla mer träningsdata"
        ])
    
    if analysis['convergence_rate'] < 0.01:
        recommendations.extend([
            "Långsam konvergens. Prova:",
            "- Öka learning rate",
            "- Minska warmup steps",
            "- Öka batch size"
        ])
    elif analysis['convergence_rate'] > 0.1:
        recommendations.extend([
            "Snabb/instabil konvergens. Prova:",
            "- Minska learning rate",
            "- Öka warmup steps",
            "- Minska batch size"
        ])
    
    return analysis, recommendations

def main():
    model_dir = "./models/finetuned-swedish-gpt"
    metrics = load_metrics(model_dir)
    
    # Plotta resultat
    plot_learning_curves(metrics, save_dir=model_dir)
    
    # Analysera träning
    analysis, recommendations = analyze_training(metrics)
    
    print("\n=== Träningsanalys ===")
    print(f"Bästa epoch: {analysis['best_epoch']}")
    print(f"Bästa valideringsförlust: {analysis['best_val_loss']:.4f}")
    print(f"Slutlig träningsförlust: {analysis['final_train_loss']:.4f}")
    print(f"Slutlig valideringsförlust: {analysis['final_val_loss']:.4f}")
    print(f"Överanpassningspoäng: {analysis['overfitting_score']:.4f}")
    print(f"Konvergenshastighet: {analysis['convergence_rate']:.4f}")
    
    if recommendations:
        print("\n=== Rekommendationer ===")
        for rec in recommendations:
            print(rec)

if __name__ == "__main__":
    main() 