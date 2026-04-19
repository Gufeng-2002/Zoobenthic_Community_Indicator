#!/usr/bin/env Rscript
# pvclust_bootstrap.R
# -------------------
# Multiscale bootstrap analysis for Ward hierarchical clustering.
#
# Usage:
#   Rscript pvclust_bootstrap.R <taxa_csv> <n_clusters> <nboot> <output_csv>
#
# Arguments:
#   taxa_csv    : Path to CSV with transformed taxa (sites x taxa, row names = site IDs)
#   n_clusters  : Number of clusters (integer)
#   nboot       : Number of bootstrap replications (integer, e.g. 1000)
#   output_csv  : Path to write the per-cluster AU/BP results
#
# Outputs:
#   A CSV with columns: Cluster, AU, BP

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript pvclust_bootstrap.R <taxa_csv> <n_clusters> <nboot> <output_csv>")
}

taxa_csv   <- args[1]
n_clusters <- as.integer(args[2])
nboot      <- as.integer(args[3])
output_csv <- args[4]

# Install pvclust if not available
if (!requireNamespace("pvclust", quietly = TRUE)) {
  install.packages("pvclust", repos = "https://cloud.r-project.org", quiet = TRUE)
}
library(pvclust)

# Read the taxa matrix (sites as rows)
taxa <- read.csv(taxa_csv, row.names = 1, check.names = FALSE)
cat(sprintf("Read taxa matrix: %d sites x %d taxa\n", nrow(taxa), ncol(taxa)))

# pvclust expects variables (taxa) as rows and observations (sites) as columns,
# so we transpose.
taxa_t <- t(taxa)

# Run pvclust with Ward's D2 method (equivalent to scipy ward) and euclidean distance
set.seed(42)
pv_result <- pvclust(
  taxa_t,
  method.hclust = "ward.D2",
  method.dist   = "euclidean",
  nboot         = nboot,
  quiet         = FALSE
)

# Extract the dendrogram and cut at k clusters
hc <- pv_result$hclust
cut_labels <- cutree(hc, k = n_clusters)

# For each cluster, find the dendrogram branch (internal node) that best
# matches the set of sites in that cluster.
# The pvclust edge numbers correspond to the merge rows in the hclust object.
n_sites <- nrow(taxa)

# Build a list of site sets for each internal node
# In the hclust merge table, rows correspond to internal nodes.
# merge[i, ] has two entries: negative = leaf, positive = previous internal node.
get_node_members <- function(hc, node_idx) {
  # Recursively get all leaf indices below a given internal node
  if (node_idx < 0) {
    return(-node_idx)
  }
  left  <- hc$merge[node_idx, 1]
  right <- hc$merge[node_idx, 2]
  return(c(get_node_members(hc, left), get_node_members(hc, right)))
}

n_internal <- nrow(hc$merge)
node_members <- vector("list", n_internal)
for (i in seq_len(n_internal)) {
  node_members[[i]] <- sort(as.integer(get_node_members(hc, i)))
}

# ── Map each k-cut cluster to its subtree root node ──────────────────
# Cutting the tree at k clusters removes the top k-1 merges.
# The k subtree roots are the children of those top merges that are
# not themselves among the cut merges.
top_merges <- seq(n_internal - n_clusters + 2, n_internal)  # e.g. k=3: {n-1, n}
children <- integer(0)
for (m in top_merges) {
  children <- c(children, hc$merge[m, 1], hc$merge[m, 2])
}
# Subtree roots = children that are internal nodes but NOT in top_merges,
# plus any leaves (negative values)
subtree_roots <- children[!(children %in% top_merges)]

# Map each subtree root to its cutree cluster label
root_to_cluster <- integer(length(subtree_roots))
for (j in seq_along(subtree_roots)) {
  root <- subtree_roots[j]
  if (root < 0) {
    # Leaf node
    representative <- -root
  } else {
    # Pick the first member of the subtree
    representative <- node_members[[root]][1]
  }
  root_to_cluster[j] <- cut_labels[representative]
}

# ── Helper: detect degenerate pvclust edge (all zeros) ───────────────
is_degenerate_edge <- function(pv_result, edge_name) {
  if (!(edge_name %in% rownames(pv_result$edges))) return(TRUE)
  row <- pv_result$edges[edge_name, ]
  return(row["au"] == 0 && row["bp"] == 0 && row["se.au"] == 0 && row["se.bp"] == 0)
}

# ── Fallback: direct bootstrap proportion for degenerate edges ───────
# Resample n sites with replacement B times, re-cluster with Ward.D2,
# cut at k, and check if the target cluster appears intact.
direct_bootstrap_cluster <- function(taxa, target_sites, n_clusters, nboot_fb = 500) {
  n <- nrow(taxa)
  count <- 0
  for (b in seq_len(nboot_fb)) {
    idx <- sample(n, n, replace = TRUE)
    boot_data <- taxa[idx, , drop = FALSE]
    # Some bootstrap samples may have duplicate rows — use unique for distance
    # but keep all for clustering (standard bootstrap approach)
    d_boot <- dist(boot_data, method = "euclidean")
    hc_boot <- hclust(d_boot, method = "ward.D2")
    labels_boot <- cutree(hc_boot, k = n_clusters)
    # Map bootstrap labels back to original indices
    # For each original-index site in target_sites, find which bootstrap
    # positions it occupies and check if they all share the same label
    # AND no non-target sites share that label.
    target_positions <- which(idx %in% target_sites)
    if (length(target_positions) == 0) next
    target_labels <- labels_boot[target_positions]
    if (length(unique(target_labels)) != 1) next
    # Check that no non-target bootstrap positions share this label
    target_label <- target_labels[1]
    non_target_positions <- which(!(idx %in% target_sites))
    non_target_with_label <- sum(labels_boot[non_target_positions] == target_label)
    if (non_target_with_label == 0) {
      count <- count + 1
    }
  }
  return(count / nboot_fb)
}

# ── Build the result table ───────────────────────────────────────────
cluster_au <- data.frame(
  Cluster = integer(n_clusters),
  AU      = numeric(n_clusters),
  BP      = numeric(n_clusters),
  stringsAsFactors = FALSE
)

for (k in seq_len(n_clusters)) {
  cluster_sites <- as.integer(sort(which(cut_labels == k)))

  # Find the subtree root for this cluster
  root_idx <- which(root_to_cluster == k)
  if (length(root_idx) == 1) {
    root_node <- subtree_roots[root_idx]
  } else {
    root_node <- NA
  }

  au_val <- NA
  bp_val <- NA

  if (!is.na(root_node) && root_node > 0) {
    edge_name <- as.character(root_node)
    if (!is_degenerate_edge(pv_result, edge_name)) {
      # Normal case: use pvclust AU/BP directly
      au_val <- pv_result$edges[edge_name, "au"]
      bp_val <- pv_result$edges[edge_name, "bp"]
      cat(sprintf("Cluster %d -> node %d: AU=%.4f, BP=%.4f (pvclust)\n",
                  k, root_node, au_val, bp_val))
    } else {
      # Degenerate edge: pvclust curve fitting failed for this near-root node.
      # Fall back to direct bootstrap proportion.
      cat(sprintf("Cluster %d -> node %d: degenerate pvclust edge (AU=0, BP=0).\n", k, root_node))
      cat(sprintf("  Computing direct bootstrap proportion (500 replicates) ...\n"))
      bp_fb <- direct_bootstrap_cluster(taxa, cluster_sites, n_clusters, nboot_fb = 500)
      au_val <- bp_fb  # Use direct bootstrap as AU estimate
      bp_val <- bp_fb
      cat(sprintf("  Fallback bootstrap proportion: %.4f\n", bp_fb))
    }
  } else if (!is.na(root_node) && root_node < 0) {
    # Single-leaf cluster — trivially supported
    au_val <- 1.0
    bp_val <- 1.0
    cat(sprintf("Cluster %d -> single leaf: AU=1.0, BP=1.0\n", k))
  } else {
    cat(sprintf("Cluster %d: could not map to dendrogram node.\n", k))
  }

  cluster_au$Cluster[k] <- k
  cluster_au$AU[k]      <- au_val
  cluster_au$BP[k]      <- bp_val
}

cat("\nCluster bootstrap support:\n")
print(cluster_au)

write.csv(cluster_au, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_csv))
