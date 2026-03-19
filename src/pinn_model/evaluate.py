"""
Evaluation for Consistency-Regularized Lensiformer.

Includes:
- ROC curves and AUC scores (per-class + macro)
- Confusion matrix
- Physics visualization (k-maps, α-fields, reconstructed source)
- k-variance per class analysis
- Rotation invariance test
- Failure analysis
- MC Dropout uncertainty
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn.functional as F

from . import config


def compute_roc_auc(model, val_loader, save_path=None):
    """
    Compute ROC curves and AUC scores.
    
    One-vs-rest for each class + macro average.
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "roc_auc.png")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Computing ROC"):
            images = images.to(config.DEVICE)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # One-hot encode labels
    n_classes = config.NUM_CLASSES
    labels_onehot = np.eye(n_classes)[all_labels]
    
    # Compute ROC for each class
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    auc_scores = {}
    
    for i, (name, color) in enumerate(zip(config.CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = roc_auc
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Macro average
    macro_auc = np.mean(list(auc_scores.values()))
    auc_scores['macro'] = macro_auc
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (Macro AUC = {macro_auc:.3f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print detailed report
    preds = all_probs.argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=config.CLASS_NAMES))
    
    print(f"\nAUC Scores:")
    for name, score in auc_scores.items():
        print(f"  {name}: {score:.4f}")
    
    return auc_scores, all_probs, all_labels


def plot_confusion_matrix(all_labels, all_probs, save_path=None):
    """Plot confusion matrix."""
    save_path = save_path or os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    
    preds = all_probs.argmax(axis=1)
    cm = confusion_matrix(all_labels, preds)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)
    
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES)
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_physics(model, val_loader, save_path=None):
    """
    Visualize physics outputs: k-maps, α-fields, reconstructed source.
    Shows 2 examples per class.
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "physics_viz.png")
    
    model.eval()
    samples = {i: [] for i in range(config.NUM_CLASSES)}
    
    # Collect 2 samples per class
    with torch.no_grad():
        for images, labels in val_loader:
            for i in range(len(labels)):
                label = labels[i].item()
                if len(samples[label]) < 2:
                    img = images[i:i+1].to(config.DEVICE)
                    _, physics = model(img, return_physics=True)
                    samples[label].append({
                        'image': img.cpu(),
                        'k': physics['k'].cpu(),
                        'alpha_x': physics['alpha_x'].cpu(),
                        'alpha_y': physics['alpha_y'].cpu(),
                        'source': physics['source'].cpu(),
                    })
            
            if all(len(v) >= 2 for v in samples.values()):
                break
    
    # Plot: rows=classes, cols=[image, k-map, α_x, α_y, source]
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    col_titles = ['Input Image', 'k-map', 'α_x (deflection)', 'α_y (deflection)', 'Reconstructed Source']
    
    for row in range(6):
        class_idx = row // 2
        sample_idx = row % 2
        data = samples[class_idx][sample_idx]
        
        axes[row, 0].imshow(data['image'][0, 0], cmap='hot')
        axes[row, 1].imshow(data['k'][0, 0], cmap='RdBu_r')
        axes[row, 2].imshow(data['alpha_x'][0, 0], cmap='RdBu_r')
        axes[row, 3].imshow(data['alpha_y'][0, 0], cmap='RdBu_r')
        axes[row, 4].imshow(data['source'][0, 0], cmap='hot')
        
        for col in range(5):
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12)
        
        axes[row, 0].set_ylabel(config.CLASS_NAMES[class_idx], fontsize=12, rotation=90)
    
    plt.suptitle('Physics Visualization per Class', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Physics visualization saved to {save_path}")


def analyze_k_variance(model, val_loader, save_path=None):
    """
    Quantitative analysis: variance of k-map per class.
    
    Expected: var(k) for vort > sphere > no
    (more substructure → more k variation)
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "k_variance.png")
    
    model.eval()
    k_variances = {name: [] for name in config.CLASS_NAMES}
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Analyzing k variance"):
            images = images.to(config.DEVICE)
            _, physics = model(images, return_physics=True)
            k = physics['k']
            
            for i in range(len(labels)):
                label = labels[i].item()
                name = config.CLASS_NAMES[label]
                var = k[i].var().item()
                k_variances[name].append(var)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    means = [np.mean(k_variances[n]) for n in config.CLASS_NAMES]
    stds = [np.std(k_variances[n]) for n in config.CLASS_NAMES]
    
    bars = ax.bar(config.CLASS_NAMES, means, yerr=stds, capsize=10,
                  color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mean:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Variance of k-map', fontsize=12)
    ax.set_title('k-map Variance per Class\n(Higher = more substructure variation)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nk-map Variance per Class:")
    for name, mean, std in zip(config.CLASS_NAMES, means, stds):
        print(f"  {name}: {mean:.6f} ± {std:.6f}")
    
    return {n: np.mean(k_variances[n]) for n in config.CLASS_NAMES}


def test_rotation_invariance(model, val_loader, save_path=None):
    """
    Rotation invariance test.
    
    Rotate input → prediction should not change.
    Reports consistency rate across 4 rotations.
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "rotation_invariance.png")
    
    model.eval()
    consistent = 0
    total = 0
    
    rotation_results = {0: [], 90: [], 180: [], 270: []}
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Rotation test"):
            images = images.to(config.DEVICE)
            
            preds_all_rotations = []
            for k in range(4):
                rotated = torch.rot90(images, k, dims=[2, 3])
                logits = model(rotated)
                preds = logits.argmax(dim=1)
                preds_all_rotations.append(preds)
                
                acc = (preds == labels.to(config.DEVICE)).float().mean().item()
                rotation_results[k * 90].append(acc)
            
            # Check consistency: all rotations give same prediction
            base_preds = preds_all_rotations[0]
            for other_preds in preds_all_rotations[1:]:
                consistent += (base_preds == other_preds).sum().item()
                total += base_preds.shape[0]
    
    consistency = 100.0 * consistent / total if total > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    angles = [0, 90, 180, 270]
    accs = [np.mean(rotation_results[a]) * 100 for a in angles]
    
    ax.bar(angles, accs, width=60, color='#3498db', edgecolor='black')
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Rotation Invariance Test\nConsistency: {consistency:.1f}%', fontsize=14)
    ax.set_xticks(angles)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nRotation Invariance: {consistency:.1f}% consistent")
    for a in angles:
        print(f"  {a}°: {np.mean(rotation_results[a])*100:.1f}% accuracy")
    
    return consistency


def failure_analysis(model, val_loader, save_path=None, n_examples=8):
    """
    Show misclassified examples with their physics maps.
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "failure_analysis.png")
    
    model.eval()
    failures = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            logits, physics = model(images, return_physics=True)
            preds = logits.argmax(dim=1)
            
            wrong = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in wrong:
                if len(failures) >= n_examples:
                    break
                failures.append({
                    'image': images[idx].cpu(),
                    'true': labels[idx].item(),
                    'pred': preds[idx].item(),
                    'k': physics['k'][idx].cpu(),
                    'source': physics['source'][idx].cpu(),
                })
            
            if len(failures) >= n_examples:
                break
    
    if not failures:
        print("No misclassifications found!")
        return
    
    n = min(len(failures), n_examples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    
    for i in range(n):
        f = failures[i]
        true_name = config.CLASS_NAMES[f['true']]
        pred_name = config.CLASS_NAMES[f['pred']]
        
        axes[i, 0].imshow(f['image'][0], cmap='hot')
        axes[i, 0].set_title(f'True: {true_name}, Pred: {pred_name}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(f['k'][0], cmap='RdBu_r')
        axes[i, 1].set_title('k-map', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(f['source'][0], cmap='hot')
        axes[i, 2].set_title('Reconstructed Source', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.suptitle('Failure Analysis: Misclassified Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Failure analysis saved to {save_path}")


def full_evaluation(model, val_loader):
    """Run complete evaluation pipeline."""
    print("\n" + "=" * 60)
    print("FULL EVALUATION")
    print("=" * 60)
    
    # ROC / AUC
    auc_scores, all_probs, all_labels = compute_roc_auc(model, val_loader)
    
    # Confusion matrix
    plot_confusion_matrix(all_labels, all_probs)
    
    # Physics analysis (only for models with physics)
    try:
        visualize_physics(model, val_loader)
        analyze_k_variance(model, val_loader)
        test_rotation_invariance(model, val_loader)
        failure_analysis(model, val_loader)
    except Exception as e:
        print(f"  Physics analysis skipped: {e}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return auc_scores
