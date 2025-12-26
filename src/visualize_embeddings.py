#!/usr/bin/env python3
"""
2D/3D Embedding Visualization for AIMO 3

Creates beautiful interactive visualizations of embeddings using:
- UMAP for dimensionality reduction
- Plotly for 2D and 3D interactive visualizations

Usage:
    python visualize_embeddings.py --input embeddings.parquet --output viz.html
    python visualize_embeddings.py --input embeddings.parquet --3d --output viz_3d.html
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

import numpy as np
import pandas as pd

# Visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("❌ Plotly not found. Install with: pip install plotly")
    sys.exit(1)

try:
    import umap
except ImportError:
    print("❌ UMAP not found. Install with: pip install umap-learn")
    sys.exit(1)


class EmbeddingVisualizer:
    """
    Interactive embedding visualizer using UMAP + Plotly.
    
    Supports:
    - 2D and 3D projections
    - Interactive HTML output
    - Multiple color schemes
    - Hover information
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def load_embeddings(self, filepath: str) -> tuple:
        """
        Load embeddings from various file formats.
        
        Args:
            filepath: Path to embeddings file (parquet, json, or npy)
        
        Returns:
            Tuple of (embeddings array, metadata dict, ids list)
        """
        filepath = Path(filepath)
        
        print(f"📂 Loading embeddings from: {filepath}")
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
            
            # Extract embeddings
            if 'embeddings' in df.columns:
                embeddings = np.array(df['embeddings'].tolist())
            elif 'embedding' in df.columns:
                embeddings = np.array(df['embedding'].tolist())
            else:
                raise ValueError("No 'embeddings' column found in parquet file")
            
            # Get IDs
            if 'id' in df.columns:
                ids = df['id'].tolist()
            else:
                ids = list(range(len(df)))
            
            # Get any additional columns as metadata
            metadata = {col: df[col].tolist() for col in df.columns 
                       if col not in ['embeddings', 'embedding', 'id']}
            
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            embeddings = np.array([item['embedding'] for item in data['embeddings']])
            ids = [item['id'] for item in data['embeddings']]
            metadata = data.get('metadata', {})
            
        elif filepath.suffix == '.npy':
            embeddings = np.load(filepath)
            
            # Try to load IDs
            ids_path = filepath.with_suffix('.ids.npy')
            if ids_path.exists():
                ids = np.load(ids_path).tolist()
            else:
                ids = list(range(len(embeddings)))
            
            metadata = {}
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"✅ Loaded {len(embeddings):,} embeddings")
        print(f"   Dimension: {embeddings.shape[1]}")
        
        return embeddings, metadata, ids
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP.
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Target dimensions (2 or 3)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric
        
        Returns:
            Reduced embeddings
        """
        print(f"\n🔮 Running UMAP dimensionality reduction...")
        print(f"   Input shape: {embeddings.shape}")
        print(f"   Output dimensions: {n_components}")
        print(f"   n_neighbors: {n_neighbors}, min_dist: {min_dist}")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state
        )
        
        reduced = reducer.fit_transform(embeddings)
        
        print(f"✅ UMAP complete! Output shape: {reduced.shape}")
        
        return reduced
    
    def create_2d_visualization(
        self,
        reduced_embeddings: np.ndarray,
        ids: List,
        labels: Optional[List] = None,
        texts: Optional[List] = None,
        title: str = "AIMO 3 Embedding Visualization (2D)",
        color_scheme: str = "Viridis"
    ) -> go.Figure:
        """
        Create an interactive 2D scatter plot.
        
        Args:
            reduced_embeddings: 2D coordinates
            ids: Record IDs
            labels: Optional labels for coloring
            texts: Optional text for hover
            title: Plot title
            color_scheme: Plotly color scheme
        
        Returns:
            Plotly Figure
        """
        print("\n🎨 Creating 2D visualization...")
        
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'id': ids
        })
        
        if labels is not None:
            df['label'] = labels
            color_col = 'label'
        else:
            df['label'] = [f"Point {i}" for i in range(len(ids))]
            color_col = None
        
        if texts is not None:
            df['text'] = [str(t)[:200] + "..." if len(str(t)) > 200 else str(t) 
                        for t in texts]
        else:
            df['text'] = df['id'].astype(str)
        
        # Create figure
        if color_col:
            fig = px.scatter(
                df, x='x', y='y',
                color=color_col,
                hover_data=['id', 'text'],
                title=title,
                color_continuous_scale=color_scheme,
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
            )
        else:
            fig = px.scatter(
                df, x='x', y='y',
                hover_data=['id', 'text'],
                title=title,
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
            )
        
        # Style the plot
        fig.update_traces(
            marker=dict(
                size=8,
                line=dict(width=0.5, color='white'),
                opacity=0.7
            )
        )
        
        fig.update_layout(
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#16213e',
            font=dict(color='white', size=12),
            title=dict(font=dict(size=20)),
            width=1200,
            height=800,
            hovermode='closest'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        )
        
        print(f"✅ 2D visualization created ({len(df):,} points)")
        
        return fig
    
    def create_3d_visualization(
        self,
        reduced_embeddings: np.ndarray,
        ids: List,
        labels: Optional[List] = None,
        texts: Optional[List] = None,
        title: str = "AIMO 3 Embedding Visualization (3D)",
        color_scheme: str = "Viridis"
    ) -> go.Figure:
        """
        Create an interactive 3D scatter plot.
        
        Args:
            reduced_embeddings: 3D coordinates
            ids: Record IDs
            labels: Optional labels for coloring
            texts: Optional text for hover
            title: Plot title
            color_scheme: Plotly color scheme
        
        Returns:
            Plotly Figure
        """
        print("\n🎨 Creating 3D visualization...")
        
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'id': ids
        })
        
        if labels is not None:
            df['label'] = labels
            color_col = 'label'
        else:
            df['label'] = [f"Point {i}" for i in range(len(ids))]
            color_col = None
        
        if texts is not None:
            df['text'] = [str(t)[:200] + "..." if len(str(t)) > 200 else str(t) 
                        for t in texts]
        else:
            df['text'] = df['id'].astype(str)
        
        # Create figure
        if color_col:
            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color=color_col,
                hover_data=['id', 'text'],
                title=title,
                color_continuous_scale=color_scheme,
                labels={
                    'x': 'UMAP Dim 1',
                    'y': 'UMAP Dim 2',
                    'z': 'UMAP Dim 3'
                }
            )
        else:
            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                hover_data=['id', 'text'],
                title=title,
                labels={
                    'x': 'UMAP Dim 1',
                    'y': 'UMAP Dim 2',
                    'z': 'UMAP Dim 3'
                }
            )
        
        # Style the plot
        fig.update_traces(
            marker=dict(
                size=4,
                line=dict(width=0.5, color='white'),
                opacity=0.8
            )
        )
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor='#1a1a2e',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                ),
                yaxis=dict(
                    backgroundcolor='#1a1a2e',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                ),
                zaxis=dict(
                    backgroundcolor='#1a1a2e',
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=True
                )
            ),
            paper_bgcolor='#16213e',
            font=dict(color='white', size=12),
            title=dict(font=dict(size=20)),
            width=1200,
            height=900
        )
        
        print(f"✅ 3D visualization created ({len(df):,} points)")
        
        return fig
    
    def create_combined_visualization(
        self,
        embeddings: np.ndarray,
        ids: List,
        labels: Optional[List] = None,
        texts: Optional[List] = None,
        title: str = "AIMO 3 Embedding Analysis"
    ) -> go.Figure:
        """
        Create a combined visualization with both 2D and 3D views.
        """
        print("\n🎨 Creating combined 2D + 3D visualization...")
        
        # Get 2D and 3D projections
        reduced_2d = self.reduce_dimensions(embeddings, n_components=2)
        reduced_3d = self.reduce_dimensions(embeddings, n_components=3)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
            subplot_titles=["2D UMAP Projection", "3D UMAP Projection"],
            horizontal_spacing=0.05
        )
        
        # Add 2D scatter
        fig.add_trace(
            go.Scatter(
                x=reduced_2d[:, 0],
                y=reduced_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=labels if labels else list(range(len(ids))),
                    colorscale='Viridis',
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"ID: {ids[i]}" for i in range(len(ids))],
                hoverinfo='text',
                name='2D'
            ),
            row=1, col=1
        )
        
        # Add 3D scatter
        fig.add_trace(
            go.Scatter3d(
                x=reduced_3d[:, 0],
                y=reduced_3d[:, 1],
                z=reduced_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=labels if labels else list(range(len(ids))),
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"ID: {ids[i]}" for i in range(len(ids))],
                hoverinfo='text',
                name='3D'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=24)),
            paper_bgcolor='#16213e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='white'),
            width=1600,
            height=800,
            showlegend=False
        )
        
        print(f"✅ Combined visualization created")
        
        return fig
    
    def save_visualization(self, fig: go.Figure, output_path: str):
        """Save visualization to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.html':
            fig.write_html(str(output_path), include_plotlyjs=True)
        elif output_path.suffix == '.png':
            fig.write_image(str(output_path), scale=2)
        elif output_path.suffix == '.pdf':
            fig.write_image(str(output_path))
        else:
            # Default to HTML
            fig.write_html(str(output_path) + '.html', include_plotlyjs=True)
        
        print(f"💾 Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 2D/3D visualizations of embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 2D visualization
  python visualize_embeddings.py --input embeddings.parquet --output viz.html
  
  # Create 3D visualization
  python visualize_embeddings.py --input embeddings.parquet --output viz_3d.html --3d
  
  # Create both 2D and 3D combined
  python visualize_embeddings.py --input embeddings.parquet --output viz.html --combined
  
  # Customize UMAP parameters
  python visualize_embeddings.py --input emb.parquet -o viz.html --n-neighbors 30 --min-dist 0.05
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input embeddings file')
    parser.add_argument('--output', '-o', required=True, help='Output visualization file')
    parser.add_argument('--3d', action='store_true', dest='use_3d', help='Create 3D visualization')
    parser.add_argument('--combined', action='store_true', help='Create combined 2D + 3D visualization')
    parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--title', default='AIMO 3 Embedding Visualization', help='Plot title')
    parser.add_argument('--color-scheme', default='Viridis', help='Plotly color scheme')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎨 AIMO 3 Embedding Visualizer")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer()
    
    # Load embeddings
    embeddings, metadata, ids = visualizer.load_embeddings(args.input)
    
    # Get labels and texts if available
    labels = metadata.get('label', metadata.get('labels', None))
    texts = metadata.get('text', metadata.get('problem', None))
    
    if args.combined:
        # Create combined visualization
        fig = visualizer.create_combined_visualization(
            embeddings, ids, labels, texts, args.title
        )
    else:
        # Reduce dimensions
        n_components = 3 if args.use_3d else 2
        reduced = visualizer.reduce_dimensions(
            embeddings,
            n_components=n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist
        )
        
        # Create visualization
        if args.use_3d:
            fig = visualizer.create_3d_visualization(
                reduced, ids, labels, texts,
                title=args.title,
                color_scheme=args.color_scheme
            )
        else:
            fig = visualizer.create_2d_visualization(
                reduced, ids, labels, texts,
                title=args.title,
                color_scheme=args.color_scheme
            )
    
    # Save visualization
    visualizer.save_visualization(fig, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

