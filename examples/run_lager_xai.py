import numpy as np
import itertools
import torch

from ANIDSC.data_source import PacketReader, CSVReader

from ANIDSC import models, feature_extractors
from ANIDSC.base_files import evaluator, MultilayerSplitter
from ANIDSC.normalizer import LivePercentile
from ANIDSC.templates import get_pipeline

from ANIDSC.models.gnnids import HomoGraphRepresentation, NodeEncoderWrapper

import warnings
import os
import yaml

METRICS = [
    "detection_rate",
    "lower_quartile_score",
    "upper_quartile_score",
    "soft_min_score",
    "soft_max_score",
    "median_score",
    "median_threshold",
    "pos_count",
    "batch_size",
]

# Setup workflow using yaml configuration file
def configure_workflow(file="config.yaml"):
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    return config

#----------------------------------------------------------------------
# ExplainEvaluator Class
#----------------------------------------------------------------------
from typing import Any, Dict, List
from networkx.drawing.nx_pydot import write_dot
import networkx as nx
import numpy as np
from ANIDSC.base_files.pipeline import PipelineComponent
from ANIDSC import evaluations 
from pathlib import Path
import time
from ANIDSC.utils import draw_graph, fig_to_array
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics.pairwise import cosine_similarity
from lime import lime_tabular
import shap

class ExplainEvaluator(PipelineComponent): 
    def __init__(self, model, prior, explain_interval, draw_graph_rep_interval, save_results:bool=True):
        """XAI evaluator that explains feature importance for anomalous outpus of a single model
        Args:
            model: The model used to calculate anomaly scores
            prior: The prior distribution used in the models loss function
            save_results (bool, optional): whether to save results in CSV file. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            draw_graph_rep_interval (bool, optional): whether to draw the graph representation. only available if pipeline contains graph representation. Defaults to False.
            explain_interval: whether to run an explanation or not
            write_header: if save_results is True, write_header will write the column names of the CSV
        """        
        super().__init__()
        self.model = model
        self.prior = prior
        self.save_results=save_results
        self.draw_graph_rep_interval=draw_graph_rep_interval
        self.explain_interval=explain_interval
        self.write_header=True
        
    def setup(self):
        super().setup()
        context=self.get_context()

        if self.save_results:
            # file to store outputs
            feature_importance_path = Path(
                f"{context['dataset_name']}/{context['fe_name']}/explanations/{context['file_name']}/{context['pipeline_name']}.csv"
            )
            feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(feature_importance_path), "w")

            # folder to dot folder
            self.dot_folder=Path(f"{context['dataset_name']}/{context['fe_name']}/dot/{context['file_name']}/{context['pipeline_name']}")
            self.dot_folder.mkdir(parents=True, exist_ok=True)
            print(f"dot folder {str(self.dot_folder)}")
        
    def teardown(self):
        if self.save_results:
            self.output_file.close()
            print("explanations file saved at", self.output_file.name)
            self.write_header=True
    
    def process(self, results:Dict[str, Any])->Dict[str, Any]:
        """processes results and log them accordingly

        Args:
            results (Dict[str, Any]): gets metric values in metric_list based on results

        Returns:
            Dict[str, Any]: dictionary of metric, value pair
        """        
        context = self.get_context()       

        result_dict = {}  # Initialise dictionary to pass on to MultilayerSplitter/CollateExplainer

        # If there is concept_drift_detection enabled (AKA Arcus or DriftSense), add these
        if context.get('concept_drift_detection',False):
            result_dict['model_idx']= results['model_idx']
            result_dict['num_model']=results['num_model']

        context = self.get_context() # Not sure why this is necessary (see line 299), but this pattern is in inherited code and I don't want to change it
    
        if context.get('G',False) and self.draw_graph_rep_interval and results['batch_num']%self.draw_graph_rep_interval==0:
            G=context['G']
            
            if results["score"] is None:
                results['score']=[0. for _ in G.nodes()]
                results['threshold']=0.
            
            nx.set_node_attributes(G, { n: {"score": score} for n, score in zip(G.nodes(), results["score"]) })
            
            malicious_threshold = results["threshold"]
            G.graph["graph"]={'threshold':malicious_threshold}

            # Run Explainability Framework every 1000 batches (currently, draw_graph_rep_interval needs to divide explain_interval)
            if results['batch_num'] % self.explain_interval == 0:

                # Setup headers for saving to CSV
                if self.save_results:
                    header = ["batch_num", "node_idx"]
                    for i in range(1,16):
                        header.append(f"lime_dim{i}")
                    for i in range(1,16):
                        header.append(f"shap_dim{i}")

                    header.append("cosine_similarity")

                # Setup values to write to CSV
                values = [results["batch_num"]]
                node_explanations = {}
                for node in G.nodes():
                    node_idx = G.nodes[node]['idx']
                    embedded_features = G.nodes[node]['x']
                    score = G.nodes[node]['score']

                    node_explanations[f"{node_idx}"] = {}

                    # Only explain malicious examples
                    if score > malicious_threshold:
                        lime_values = self.get_lime(np.array(embedded_features))
                        lime_norm = self.normalise(lime_values)

                        shap_values = self.get_shap(np.array(embedded_features))
                        shap_norm = self.normalise(shap_values)

                        similarity_score = cosine_similarity(
                            shap_norm.reshape(1, -1), 
                            lime_norm.reshape(1, -1)
                        )[0][0]

                        for i in range(1, len(lime_norm)+1):
                            node_explanations[f"{node_idx}"][f"lime_dim{i}"] = lime_values[i - 1]
                        for i in range(1, len(shap_norm)+1):
                            node_explanations[f"{node_idx}"][f"shap_dim{i}"] = shap_values[i - 1]

                        node_explanations[f"{node_idx}"]["similarity_score"] = similarity_score                         

                        # Update Graph for writing to dot
                        G.nodes[node]['shap'] = ",".join(map(str, shap_values))
                        G.nodes[node]['lime'] = lime_values
                        G.nodes[node]['shap_norm'] = ",".join(map(str, shap_norm))
                        G.nodes[node]['lime_norm'] = lime_norm
                        G.nodes[node]['similarity_score'] = similarity_score

                        # Only run GNNExplainer if normalised shap and lime values are similar
                        if similarity_score > 0.99:
                            continue
                            # feature_mask, edge_mask = self.get_gnnx(G, node)
                            # G.nodes[node]['feature_mask'] = node_mask
                            # G.ndoes[node]['edge_mask'] = edge_mask
            
                result_dict["node_explanations"] = node_explanations
            result_dict["G"]=G
        
        result_dict['batch_num']=results['batch_num']
        return result_dict

    # Wrapper function for explainability modules
    def predict(self, embedding):
        tensor = torch.from_numpy(embedding).float().to(self.model.device)
        results = self.model.process(tensor)
        scores = results["score"]

        return np.array(scores)

    # Returns LIME values on embedding dimensions on a single node
    def get_lime(self, embedding):
        explainer = self.lime_explainer
        result = explainer.explain_instance(data_row=embedding, predict_fn=self.predict, num_features=15, num_samples=2500)
        fv = result.as_list()
        lime_values = [] 
        for feature_name, lime_value in fv:
            lime_values.append(lime_value)

        return lime_values
        
    # Returns SHAP values on embedding dimensions on a single node
    def get_shap(self, embedding):
        data = shap.sample(self.prior, 50)
        explainer = shap.KernelExplainer(self.predict, data)

        return explainer.shap_values(embedding)

    # Returns a Node Mask and Edge Mask detailing which notes contributed most to embedding
    def get_gnnx(self, embedding):
        pass

    # Takes an np array as SHAP and LIME return NP arrays
    def normalise(self, x):
        x = np.array(x)
        data_range = x.max() - x.min()
        if data_range == 0:
            return np.zeros(len(x))
        return (x - x.min()) / (data_range)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lime_explainer' in state:
            del state['lime_explainer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data = self.prior, # Generate prior distribution to which the embedding was trained against
            mode = "regression", # I believe this is correct
            feature_names = ["Dim1", "Dim2", "Dim3", "Dim4", "Dim5", "Dim6", "Dim7", "Dim8", "Dim9", "Dim10", "Dim11", "Dim12", "Dim13", "Dim14", "Dim15"],
            random_state = 42
        )

class CollateExplainEvaluator(PipelineComponent): # Still gotta edit this
    def __init__(self, log_to_tensorboard:bool=True, save_results:bool=True):
        """Evaluator to aggregate results from multiple base evaluators

        Args:
            log_to_tensorboard (bool, optional): whether to log results in tensorboard. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. Defaults to True.
        """        
        super().__init__()
        self.log_to_tensorboard=log_to_tensorboard
        self.save_results=save_results
        
    def setup(self):
        context=self.get_context()
        if self.save_results:
            # file to store outputs
            layerwise_path = Path(
                f"{context['dataset_name']}/{context['fe_name']}/explanations/{context['file_name']}/{context['pipeline_name']}.csv"
            )
            layerwise_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(layerwise_path), "w")
            self.write_output_header=True 
            
            
            # folder to dot folder
            self.dot_folder=Path(f"{context['dataset_name']}/{context['fe_name']}/dot/{context['file_name']}/{context['pipeline_name']}")
            self.dot_folder.mkdir(parents=True, exist_ok=True)
            print(f"dot folder {str(self.dot_folder)}")
    
    def teardown(self):
        self.output_file.close()
        print("results file saved at", self.output_file.name)
    
    def process(self, results:Dict[str, Dict[str, Any]]):
        """process the results dictionary

        Args:
            results (Dict[str, Dict[str, Any]]): layer: results pairs for each layer
        """        
    
        for protocol, result_dict in results.items():
            if result_dict is None:
                continue
            
            if self.save_results:
                if "G" in result_dict.keys():
                    write_dot(result_dict["G"], f"{str(self.dot_folder)}/{protocol}_{result_dict['batch_num']}.dot")
                    del result_dict["G"]

                batch_num = result_dict["batch_num"]
                node_explanations = result_dict.get("node_explanations", {})

                for node_idx, explanation in node_explanations.items():
                    row = [batch_num, protocol, node_idx]

                    for i in range(1, 16):
                        row.append(explanation.get(f"lime_dim{i}", ""))
                    for i in range(1,16):
                        row.append(explanation.get(f"shap_dim{i}", ""))
                    row.append(explanation.get("similarity_score",""))
                
                
                    if self.write_output_header:
                        header = ["batch_num", "protocol", "node_idx"]
                        header += [f"lime_dim{i}" for i in range(1, 16)]
                        header += [f"shap_dim{i}" for i in range(1, 16)]
                        header.append("cosine_similarity")
                        self.output_file.write(",".join(header) + "\n")
                        self.write_output_header=False
                    
                
                    self.output_file.write(
                        ",".join(list(map(str, row))) + "\n"
                    )     
        

#----------------------------------------------------------------------
# Generate a random sample from prior distribution for LIME pertabation
def sample_prior(distribution='log-normal', examples=500, seed=42):
    prior_generator = np.random.default_rng(seed=seed)
    shape = [examples, 15] # 15 latent dimensions hard coded here 

    # Parameters adapted from models/gnnids.py - generate_z()
    match distribution:
        case 'log-normal':
            return prior_generator.lognormal(mean=0.0, sigma=1.0, size=shape)
        case 'uniform':
            return prior_generator.uniform(low=-1.0, high=1.0, size=shape)
        case 'gaussian':
            return prior_generator.normal(loc=0.0, scale=1.0, size=shape)

def extract_features(dataset, benign_file, fe_cls, fe_name):
    # Takes roughly 5-7 hours to run this function... so sit tight 🍵

    offline_reader = PacketReader(dataset, benign_file)
    pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls})
    offline_reader >> pipeline
    offline_reader.start()

    attacks = [
        "ACK_Flooding",
        #"UDP_Flooding",
        #"SYN_Flooding",
        "ARP_Spoofing",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google-Nest-Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        #"Smart_Clock_1",
        #"Smart_TV",
        "Smartphone_1",
        #"Smartphone_2",
    ]

    for d, a in itertools.product(devices, attacks):
        file_name = f"attack_samples/{d}/{a}_{d}"

        offline_reader = PacketReader(dataset, file_name)
        pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls}, load_existing=[dataset, fe_name, benign_file])
        offline_reader >> pipeline
        offline_reader.start()

def cicids_extract_features(dataset, benign_file, fe_cls, fe_name):
    offline_reader = PacketReader(dataset, benign_file)
    pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls})
    offline_reader >> pipeline
    offline_reader.start()

    mal_files = [
        "Tuesday-WorkingHours.pcap_ISCX",
        "Wednesday-workingHours-Afternoon-DDos.pcap_ISCX",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
        "Friday-WorkingHours-Morning.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",        
    ]

    for file in mal_files:
        offline_reader = PacketReader(dataset, file)
        pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls}, load_existing=[dataset, fe_name, benign_file])
        offline_reader >> pipeline
        offline_reader.start()

def test_extract_features(dataset, benign_file, fe_cls, fe_name):
    offline_reader = PacketReader(dataset, benign_file)
    pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls})
    offline_reader >> pipeline
    offline_reader.start()

    mal_files = [
        "malicious_ACK_Flooding",
        "malicious_Port_Scanning",
        "malicious_Service_Detection",  
    ]

    for file in mal_files:
        offline_reader = PacketReader(dataset, file)
        pipeline = get_pipeline(["feature_extraction"], {"fe_cls":fe_cls}, load_existing=[dataset, fe_name, benign_file])
        offline_reader >> pipeline
        offline_reader.start()
    
# Based off of get_pipeline() from templates.py
def create_pipeline(
    pipeline_desc={"fe_cls":"AfterImageGraph", "model_name":"ICL"},
    load_existing=False,
    ):

    # Copied from templates.py
    def load_or_create(class_type, folder, name=None, **kwargs):
        if load_existing:
            loaded_obj = class_type.load_pickle(folder, *load_existing, name or load_existing[1])
            if loaded_obj is not None:
                return loaded_obj
            else:
                print(f"Object not found, creating new {class_type.__name__}")
        
        return class_type(**kwargs)

    # Models for Outlier Detection
    model_class = getattr(models, pipeline_desc["model_name"])
    model = load_or_create(model_class, "models", name=pipeline_desc["model_name"], preprocessors=[], profile=False)

    model_name = f"MultlayerSplitter(LivePercentile-HomoGraphRepresentation-NodeEncoderWrapper({pipeline_desc['node_encoder']}({pipeline_desc['distribution']})-{model.__str__()})-ExplainEvaluator)"
    
    # Train LIME Explainer Models
    prior = sample_prior(distribution=pipeline_desc["distribution"], examples=500, seed=42)

    # XAI Evaluations on configurations
    xeval = ExplainEvaluator(
        model = model,
        prior=prior,
        explain_interval=1000, # explanations are expensive
        draw_graph_rep_interval=100,
        save_results=False, # pass to xcollate to save results
    )  

    # # Vanilla evaluator from original repository
    # baseval = evaluator.BaseEvaluator(
    #     METRICS,
    #     log_to_tensorboard=False,
    #     save_results=False,
    #     draw_graph_rep_interval=100,
    # )

    # Graph Encoding
    standardiser = LivePercentile()
    graph_representation = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"])
    node_encoder = getattr(models, pipeline_desc['node_encoder'])(15, pipeline_desc['distribution'])
    encoder_model = NodeEncoderWrapper(node_encoder, model)
    
    # Layer-wise Splitter
    protocol_splitter = load_or_create(
        MultilayerSplitter,
        "models",
        model_name,
        pipeline=(standardiser | graph_representation | encoder_model | xeval)
    )

    xcollate = CollateExplainEvaluator(save_results=True) # The collate_evaluator will save to output location
    # collate = evaluator.CollateEvaluator(log_to_tensorboard=False, save_results=True)
    
    # Construct Pipeline
    # pipeline = protocol_splitter | xcollate
    pipeline = protocol_splitter | collate

    return pipeline

def run_benign_pipeline(dataset, benign_file, pipeline_desc, fe_class, fe_name, batch_size):
    dist = pipeline_desc["distribution"]
    encoder = pipeline_desc["node_encoder"]
    model = pipeline_desc["model_name"]

    print(f"Creating Pipeline for {dist}-{encoder}-{model}")
    pipeline = create_pipeline(pipeline_desc)

    print("Running Pipeline...")
    offline_reader = CSVReader(
        dataset, 
        fe_class, 
        fe_name, 
        benign_file
    )
    offline_reader >> pipeline
    offline_reader.start()

def run_attack_pipeline(dataset, benign_file, pipeline_desc, fe_class, fe_name, batch_size):
    attacks = [
        "ACK_Flooding",
        #"UDP_Flooding",
        #"SYN_Flooding",
        "ARP_Spoofing",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google-Nest-Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        #"Smart_Clock_1",
        #"Smart_TV",
        "Smartphone_1",
        #"Smartphone_2",
    ]

    for d, a in itertools.product(devices, attacks):
        file_name = f"attack_samples/{d}/{a}_{d}"

        if (d == "Raspberry_Pi_telnet" or d == "Smartphone_1") and a == "ARP_Spoofing":
            continue
        else:
            offline_reader = CSVReader(
                dataset, 
                fe_class, 
                fe_name, 
                file_name,
            )

            pipeline = create_pipeline(pipeline_desc=pipeline_desc, load_existing=[dataset, fe_name, benign_file])

            offline_reader >> pipeline
            offline_reader.start()

def cicids_run_attack_pipeline(dataset, benign_file, pipeline_desc, fe_name, fe_class, batch_size):
    mal_files = [
        "Tuesday-WorkingHours.pcap_ISCX",
        "Wednesday-workingHours-Afternoon-DDos.pcap_ISCX",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
        "Friday-WorkingHours-Morning.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",        
    ]

    for file_name in mal_files:
        offline_reader = CSVReader(
            dataset, 
            fe_class, 
            fe_name, 
            file_name,
            batch_size
        )

        pipeline = create_pipeline(pipeline_desc=pipeline_desc, load_existing=[dataset, fe_name, benign_file])

        offline_reader >> pipeline
        offline_reader.start()

def test_run_attack_pipeline(dataset, benign_file, pipeline_desc, fe_name, fe_class, batch_size):
    mal_files = [
        "malicious_ACK_Flooding",
        "malicious_Port_Scanning",
        "malicious_Service_Detection",  
    ]

    for file_name in mal_files:
        offline_reader = CSVReader(
            dataset, 
            fe_name, 
            fe_class,
            file_name,
            batch_size
        )

        pipeline = create_pipeline(pipeline_desc=pipeline_desc, load_existing=[dataset, fe_name, benign_file])

        offline_reader >> pipeline
        offline_reader.start()

def get_files(dataset, fe_name):
    base = dataset + f"/{fe_name}/feature_extractors/" 
                
    benign_file = "benign_samples/whole_week"
    attacks = [
        "ACK_Flooding",
        #"UDP_Flooding",
        #"SYN_Flooding",
        "ARP_Spoofing",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google-Nest-Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        #"Smart_Clock_1",
        #"Smart_TV",
        "Smartphone_1",
        #"Smartphone_2",
    ]

    files = [base + f"benign_samples/whole_week/{fe_name}.pkl"]

    for d, a in itertools.product(devices, attacks):
        files.append(f"{base}/attack_samples/{d}/{a}_{d}/{fe_name}.pkl")

    return files

def get_cicids_files(dataset, fe_name):
    base = dataset + f"/{fe_name}/feature_extractors/"
    files = [
        "Monday-WorkingHours.pcap_ISCX",
        "Tuesday-WorkingHours.pcap_ISCX",
        "Wednesday-workingHours-Afternoon-DDos.pcap_ISCX",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
        "Friday-WorkingHours-Morning.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",        
    ]

    checks = []
    for f in files:
        checks.append(f"{base}{f}/{fe_name}.pkl")

    return checks

def get_test_files(dataset, fe_name):
    base = dataset + f"/{fe_name}/feature_extractors/"
    files = [
        "benign_lenovo_bulb",
        "malicious_ACK_Flooding",
        "malicious_Port_Scanning",
        "malicious_Service_Detection",  
    ]
    
    checks = []
    for f in files:
        checks.append(f"{base}{f}/{fe_name}.pkl")

    return checks

# Check for the existence of extracted features
def fe_pkls_exist(dataset, fe_name):
    files = get_files(dataset, fe_name)
    for f in files:
        if not os.path.exists(f):
            print(f"Not found: {f}")
            return True
    
    return False

# Check for the existence of extracted features
def cicids_fe_pkls_exist(dataset, fe_name):
    files = get_cicids_files(dataset, fe_name)    
    for f in files:
        if not os.path.exists(f):
            print(f"Not found: {f}")
            return True
    
    return False

def test_fe_pkls_exist(dataset, fe_name):
    files = get_test_files(dataset, fe_name) 
    for f in files:
        if not os.path.exists(f):
            print(f"Not found: {f}")
            return True
    
    return False

if __name__=="__main__":
    configs = "../configs"

    for config in os.listdir(configs):
        print(f"CONFIGURATION: {config}")
        print("------------------------------------")

        # Load the configuration file
        settings = configure_workflow(os.path.join(configs, config))

        # Skip if configured to do so
        if settings.get('skip', False):
            print("skipping...")
            print("------------------------------------")
            continue

        # Set all the variablse
        dataset = settings.get('dataset', '../datasets/')
        batch_size = settings.get('batch_size', 256)
        benign_file = settings.get('benign_file', None)
        fe_class = settings.get('fe_class', 'AfterImage')
        fe_name = settings.get('fe_name', 'AfterImage')
        model_names = settings.get('model_names', ['AE'])
        node_encoders = settings.get('node_encoders', ['LinearNodeEncoder', 'GCNNodeEncoder', 'GATNodeEncoder'])
        distribution = settings.get('distribution', ['log-normal', 'uniform', 'gaussian'])

        # Set functions based on dataset 
        if dataset == "../datasets/UQIOT2022":
            fe_exists = fe_pkls_exist
            extractor = extract_features
            run_attack = run_attack_pipeline
        elif dataset == "../datasets/CICIDS-2017":
            fe_exists = cicids_fe_pkls_exist
            extractor = cicids_extract_features
            run_attack = cicids_run_attack_pipeline
        elif dataset == "../datasets/test_data":
            fe_exists = test_fe_pkls_exist
            extractor = test_extract_features
            run_attack = test_run_attack_pipeline
        else:
            print("Dataset path contains no data. Check that path is correct")
    
        # Extract Features for reuse in pipeline (`fe` stands for `feature_extractor`)
        if fe_exists(dataset, fe_name):
            print(f"extracting features with {fe_name}...")
            extractor(dataset, benign_file, fe_class, fe_name)  
            print("feature extraction done")

        # Run the configurations and product output files
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            for dist, encoder, model in itertools.product(distribution, node_encoders, model_names):
        
                pipeline_desc = {
                    "fe_cls": fe_class,
                    "model_name" : model,
                    "node_encoder" : encoder,
                    "distribution" : dist,
                }

                run_benign_pipeline(dataset, benign_file, pipeline_desc, fe_class, fe_name, batch_size)
                run_attack(dataset, benign_file, pipeline_desc, fe_class, fe_name, batch_size)
        

        print("------------------------------------")

