def plot_node(ax, text, xy, xytext, arrowprops=None, bbox_props=None, fontsize=10):
        ax.annotate(text, xy=xy, xytext=xytext,
                    arrowprops=arrowprops or dict(facecolor='black', arrowstyle='->'),
                    bbox=bbox_props or dict(boxstyle="round,pad=0.5", fc="cyan", ec="b", lw=1),
                    horizontalalignment='center', verticalalignment='center', fontsize=fontsize)

def create_tree_diagram(cut_points_list, num_classes):
    fig, ax = plt.subplots(figsize=(15, 6))  # Increased figure size for more horizontal space
    max_layer_width = max(len(cut_points) for cut_points in cut_points_list) * 1.5
    ax.set_xlim(-2, max_layer_width)  # Extended x-axis limits
    ax.set_ylim(-2, len(cut_points_list) + 2)  # Extended y-axis limits for additional space
    ax.axis('off')

    # Increased font size for readability
    plot_node(ax, 'Root', (max_layer_width / 2, len(cut_points_list) + 2), (max_layer_width / 2, len(cut_points_list) + 2.5), fontsize=12)

    current_layer = [(max_layer_width / 2, len(cut_points_list) + 2)]
    for layer_index, cut_points in enumerate(cut_points_list):
        next_layer = []
        layer_y = len(cut_points_list) - layer_index + 1
        # Adjusted node_gap to increase space exponentially with tree depth
        node_gap = max_layer_width / (2 ** (layer_index + 1))
        for node_x, _ in current_layer:
            for i, cut_point in enumerate(cut_points):
                rounded_cut_point = round(cut_point.item(), 4)
                node_text = f'F{layer_index+1} <= {rounded_cut_point}'
                child_x = node_x - (node_gap * (len(cut_points) - 1)) + i * (node_gap * 2)
                next_layer.append((child_x, layer_y))
                plot_node(ax, node_text, (child_x, layer_y + 0.5), (node_x, layer_y + 1), fontsize=10)
        current_layer = next_layer

    # Plotting leaf nodes with increased gap
    leaf_gap = max_layer_width / (2 ** (len(cut_points_list) + 1))
    for i, (leaf_x, _) in enumerate(current_layer):
        class_label = f'Class {i % num_classes + 1}'
        leaf_text = f'Leaf {i+1}\n{class_label}'
        plot_node(ax, leaf_text, (leaf_x, 0), (leaf_x, 1),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox_props=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="b", lw=1), fontsize=10)

    plt.show()

# Example cut points - replace with actual cut points from your model
create_tree_diagram(cut_points_list, num_classes)