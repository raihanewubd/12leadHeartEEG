import numpy as np
import torch
from torch.utils.data import Dataset
import random
import streamlit as st
import dill as pickle
import plotly.graph_objects as go

# Import your model class
from model_file import ECGLeadNet  # Adjust import/path as needed

# Set a random seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

@st.cache_resource
def load_model():
    model = ECGLeadNet()
    model.load_state_dict(
        torch.load("everyEpochModelv1.pth", map_location=torch.device('cpu'))
    )
    model.eval()
    return model

@st.cache_resource
def load_dataset_from_pickle(pickle_path: str):
    """
    Load dataset from a pickle file and return the dataset along with loading time.
    """
    import time  # Kept here for timing; used in the function only
    start_time = time.time()
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    end_time = time.time()
    loading_time = end_time - start_time
    return dataset, loading_time

# Set page layout to 'wide'
st.set_page_config(layout="wide")

def visualize_raw_input(col, data_np, title="Raw ECG (12, 5000)"):
    """
    Visualize the raw ECG input data using Plotly.
    data_np should have shape (12, 5000).
    Displays in the given Streamlit column (col).
    """
    fig = go.Figure()
    for i in range(data_np.shape[0]):
        fig.add_trace(
            go.Scatter(
                y=data_np[i],
                mode='lines',
                name=f'Lead {i + 1}'
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    col.plotly_chart(fig)

def visualize_layer_output(col, layer_name, layer_output):
    """
    Visualize a single layer's output tensor with Plotly in the given column.
    layer_output shape: [batch_size, num_channels, length].
    """
    col.write(f"**Layer**: {layer_name}, Output shape: {list(layer_output.shape)}")

    # Remove batch dimension if batch_size=1 => [channels, length]
    layer_output_np = layer_output[0].cpu().numpy()
    num_channels, length = layer_output_np.shape

    fig = go.Figure()
    for ch in range(num_channels):
        fig.add_trace(
            go.Scatter(
                y=layer_output_np[ch],
                mode='lines',
                name=f'Channel {ch + 1}'
            )
        )
    fig.update_layout(
        title=f"Feature Map: {layer_name}",
        xaxis_title="Time/Index",
        yaxis_title="Activation",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    col.plotly_chart(fig)

def visualize_2d_output(col, layer_name, layer_output):
    """
    Visualize 2D output ([batch_size, features]) as a bar chart.
    """
    col.write(f"**Layer**: {layer_name}, Output shape: {list(layer_output.shape)}")
    arr_1d = layer_output[0].cpu().numpy().astype(float)
    col.bar_chart(arr_1d)

class SimpleDataset(Dataset):
    def __init__(self, data_list, label_list, original_indices_list):
        self.data_list = data_list
        self.label_list = label_list
        self.original_indices_list = original_indices_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.data_list[i], self.label_list[i], self.original_indices_list[i]

def main():
    st.title("ECG Feature Map Visualization (Wide Mode)")

    # Load the model
    model = load_model()

    # If not already loaded, load dataset
    if 'dataset' not in st.session_state or st.session_state['dataset'] is None:
        dataset, loading_time = load_dataset_from_pickle('physionet_subset_dataset_dill.pkl')
        st.session_state['dataset'] = dataset
        st.write(f"Dataset loaded successfully in {loading_time:.4f} seconds.")
        st.write(f"Total dataset size: **{len(dataset)}**")

    dataset = st.session_state['dataset']
    dataset_size = len(dataset)
    
    st.write(f"**Dataset size**: {dataset_size}")
    
    # Let user pick which sample to visualize
    sample_index = st.number_input(
        "Choose sample index",
        min_value=0,
        max_value=dataset_size - 1,
        value=0,
        step=1
    )
    
    # Retrieve that specific sample
    ecg_tensor, label_tensor, orig_idx = dataset[sample_index]
    ecg_data = ecg_tensor.cpu().numpy()  # shape => (12, 5000) if stored that way
    
    st.write(f"**Original index**: {orig_idx} | **Label**: {label_tensor.item()}")

    # Create two columns side by side
    col_left, col_right = st.columns([1, 1])

    # ------------ LEFT COLUMN ------------
    with col_left:
        st.markdown("#### Raw Input ECG")
        visualize_raw_input(col_left, ecg_data, title=f"Raw ECG (Sample Index {sample_index})")

    # ------------ RIGHT COLUMN ------------
    with col_right:
        st.markdown(
            """
            <div style="height:500px; overflow-y:scroll; border: 1px solid #ccc; padding: 10px;">
            """,
            unsafe_allow_html=True
        )
        # Forward pass to get layer outputs
        ecg_batch = ecg_tensor.unsqueeze(0)  # shape => [1, 12, 5000]
        model.eval()
        with torch.no_grad():
            internal_outputs = model(ecg_batch)

        # Show predicted output
        final_logits = internal_outputs['fc2']
        predicted_class = torch.argmax(final_logits, dim=1).item()
        st.write(f"**Predicted class**: {predicted_class}")

        # Show layer selection checkboxes
        st.markdown("### Select Layers to Visualize")

        # Single "View All" checkbox
        view_all_flag = st.checkbox("View ALL Layers", value=False)

        layer_names = list(internal_outputs.keys())
        user_checks = {}
        for layer_name in layer_names:
            user_checks[layer_name] = st.checkbox(f"{layer_name}", value=False)

        # Visualization region
        with st.expander("Visualizations", expanded=True):
            for layer_name in layer_names:
                if view_all_flag or user_checks[layer_name]:
                    layer_output = internal_outputs[layer_name]
                    if layer_output.ndim == 3:
                        visualize_layer_output(st, layer_name, layer_output)
                    elif layer_output.ndim == 2:
                        visualize_2d_output(st, layer_name, layer_output)
                    else:
                        st.write(
                            f"**Layer**: {layer_name}, shape={list(layer_output.shape)} "
                            "- cannot visualize directly."
                        )
        st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
