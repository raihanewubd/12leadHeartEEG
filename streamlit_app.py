import numpy as np
import torch
import streamlit as st
import dill as pickle
import plotly.graph_objects as go
import random

# If you have a separate file for your model:
from model_file import ECGLeadNet  # Adjust if needed

# -------------------------
# Set random seeds
# -------------------------
seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

st.set_page_config(layout="wide")

# -------------------------
# Caching / Loading
# -------------------------
@st.cache_resource
def load_model(model_path="everyEpochModelv1.pth"):
    model = ECGLeadNet()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    model.eval()
    return model

@st.cache_resource
def load_dataset_from_pickle(pickle_path: str):
    import time
    start_time = time.time()
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    end_time = time.time()
    loading_time = end_time - start_time
    return dataset, loading_time

# -------------------------
# Visualization Functions
# -------------------------
def visualize_raw_input(col, data_np, title="Raw ECG (12, 5000)"):
    """
    Visualize the raw ECG input data using Plotly in the specified column.
    data_np shape => (12, 5000)
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
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=-0.2,
        #     xanchor="center",
        #     x=0.5
        # ),
        # width=400,
        # height=300
    )
    col.plotly_chart(fig)

def visualize_layer_output(col, layer_name, layer_output):
    """
    Visualize a single layer's 3D output [1, channels, length] as multiple line plots.
    """
    col.write(f"**Layer**: {layer_name}, Output shape: {list(layer_output.shape)}")

    # remove batch dimension => (channels, length)
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
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=-0.2,
        #     xanchor="center",
        #     x=0.5
        # ),
        # width=600,
        # height=300
    )
    col.plotly_chart(fig)

def visualize_2d_output(col, layer_name, layer_output):
    """
    Visualize a single layer's 2D output [1, features] as a bar chart.
    """
    col.write(f"**Layer**: {layer_name}, Output shape: {list(layer_output.shape)}")
    arr_1d = layer_output[0].cpu().numpy().astype(float)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(arr_1d))),
            y=arr_1d,
            name=layer_name
        )
    )
    fig.update_layout(
        title=f"{layer_name} (2D)",
        xaxis_title="Feature Index",
        yaxis_title="Value",
        template="plotly_dark",
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=-0.2,
        #     xanchor="center",
        #     x=0.5
        # ),
        # width=600,
        # height=300
    )
    col.plotly_chart(fig)

# -------------------------
# SimpleDataset (if needed)
# -------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list, original_indices_list):
        self.data_list = data_list
        self.label_list = label_list
        self.original_indices_list = original_indices_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx], self.original_indices_list[idx]


# -------------------------
# Main App
# -------------------------
def main():
    st.title("ECG Visualization: Left = Raw Signal, Right = Layer Outputs")

    # Load model if not in session_state
    if 'model' not in st.session_state:
        st.session_state['model'] = load_model("everyEpochModelv1.pth")

    # Load dataset if not in session_state
    if 'dataset' not in st.session_state:
        dataset, loading_time = load_dataset_from_pickle("physionet_subset_dataset_dill.pkl")
        st.session_state['dataset'] = dataset
        st.write(f"Dataset loaded in {loading_time:.2f} seconds. Size = {len(dataset)}")

    model = st.session_state['model']
    dataset = st.session_state['dataset']

    # Let user pick sample index
    dataset_size = len(dataset)
    sample_index = st.number_input(
        "Sample index",
        min_value=0,
        max_value=dataset_size - 1,
        value=0,
        step=1
    )

    ecg_tensor, label_tensor, orig_idx = dataset[sample_index]
    ecg_data = ecg_tensor.cpu().numpy()

    st.write(f"Original index: {orig_idx}, Label: {label_tensor.item()}")

    # Create two columns: left for raw ECG, right for layer selection & outputs
    col_left, col_right = st.columns([1,1])

    # ----- LEFT COLUMN (Raw ECG) -----
    with col_left:
        st.markdown("### Raw ECG")
        visualize_raw_input(col_left, ecg_data, title=f"Raw ECG (Index {sample_index})")

    # ----- RIGHT COLUMN (Layers) -----
    with col_right:
        st.markdown("### Model Layer Visualization")

        # Run forward pass
        model.eval()
        with torch.no_grad():
            internal_outputs = model(ecg_tensor.unsqueeze(0))  # shape => [1,12,5000]

        # Show predicted class
        final_logits = internal_outputs['fc2']
        predicted_class = torch.argmax(final_logits, dim=1).item()
        st.write(f"**Predicted class**: {predicted_class}")

        # Build list for selectbox (All Layers + each layer name)
        layer_names = list(internal_outputs.keys())
        options = ["All Layers"] + layer_names

        selected_layer = st.selectbox(
            "Select a layer to visualize (or All Layers)",
            options
        )

        # If "All Layers" => visualize them all, else just the one
        if selected_layer == "All Layers":
            st.write("#### Visualizing ALL layers")
            for ln in layer_names:
                lo = internal_outputs[ln]
                if lo.ndim == 3:       # [1, channels, length]
                    visualize_layer_output(col_right, ln, lo)
                elif lo.ndim == 2:    # [1, features]
                    visualize_2d_output(col_right, ln, lo)
                else:
                    col_right.write(f"Layer {ln} shape {list(lo.shape)} not supported.")
        else:
            # Visualize only the chosen layer
            st.write(f"#### Visualizing layer: {selected_layer}")
            lo = internal_outputs[selected_layer]
            if lo.ndim == 3:
                visualize_layer_output(col_right, selected_layer, lo)
            elif lo.ndim == 2:
                visualize_2d_output(col_right, selected_layer, lo)
            else:
                col_right.write(f"Shape {list(lo.shape)} not supported for {selected_layer}.")

if __name__ == "__main__":
    main()
