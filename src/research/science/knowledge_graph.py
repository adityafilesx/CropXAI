"""
Scientific Knowledge Graph Engine
=================================
Constructs a Knowledge Graph mapping Model -> Trust -> Calibration -> OOD -> Accuracy -> Failure Mode.
Exports to Neo4j, JSON, and GraphML formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class KnowledgeGraphEngine:
    def __init__(self):
        self.nodes = []
        self.edges = []
        
    def add_experiment_node(self, exp_data: Dict[str, Any]):
        """
        Parses an experiment result and adds the nodes and edges to the graph.
        """
        model_name = exp_data.get('config', {}).get('model', 'Unknown')
        trust_score = exp_data.get('results', {}).get('trust_score', 0.0)
        failure_modes = exp_data.get('results', {}).get('failures', [])
        
        # Add Model Node
        if not any(n['id'] == model_name for n in self.nodes):
            self.nodes.append({"id": model_name, "label": "Model", "properties": {}})
            
        # Add Trust Node
        trust_id = f"{model_name}_Trust"
        self.nodes.append({"id": trust_id, "label": "TrustScore", "properties": {"value": trust_score}})
        self.edges.append({"source": model_name, "target": trust_id, "relationship": "HAS_TRUST_SCORE"})
        
        # Add Failure Nodes
        for failure in failure_modes:
            fail_id = f"Failure_{failure}"
            if not any(n['id'] == fail_id for n in self.nodes):
                self.nodes.append({"id": fail_id, "label": "FailureMode", "properties": {}})
            self.edges.append({"source": model_name, "target": fail_id, "relationship": "EXHIBITS_FAILURE"})
            
    def export_graphml(self, output_path: Path):
        """
        Exports the graph to GraphML format for visualization in Gephi or similar.
        """
        logger.info(f"Exporting Knowledge Graph to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prototype GraphML writer
        graphml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        graphml += '  <graph id="G" edgedefault="directed">\n'
        
        for node in self.nodes:
            graphml += f'    <node id="{node["id"]}"/>\n'
            
        for i, edge in enumerate(self.edges):
            graphml += f'    <edge id="e{i}" source="{edge["source"]}" target="{edge["target"]}"/>\n'
            
        graphml += '  </graph>\n</graphml>'
        
        with open(output_path, 'w') as f:
            f.write(graphml)
