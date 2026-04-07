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
  node_members[[i]] <- sort(get_node_members(hc, i))
}

# For each cluster, find the internal node whose member set exactly matches
# (or best matches) the cluster sites.
cluster_au <- data.frame(
  Cluster = integer(n_clusters),
  AU      = numeric(n_clusters),
  BP      = numeric(n_clusters),
  stringsAsFactors = FALSE
)

for (k in seq_len(n_clusters)) {
  cluster_sites <- sort(which(cut_labels == k))

  # Try exact match first
  best_node <- NA
  for (i in seq_len(n_internal)) {
    if (identical(node_members[[i]], cluster_sites)) {
      best_node <- i
      break
    }
  }

  # If no exact match, find the smallest superset
  if (is.na(best_node)) {
    best_size <- Inf
    for (i in seq_len(n_internal)) {
      members <- node_members[[i]]
      if (all(cluster_sites %in% members) && length(members) < best_size) {
        best_size <- length(members)
        best_node <- i
      }
    }
  }

  if (!is.na(best_node)) {
    # pvclust stores AU and BP p-values in $edges with row names = edge number
    # Edge numbers correspond to internal node indices
    edge_name <- as.character(best_node)
    if (edge_name %in% rownames(pv_result$edges)) {
      au_val <- pv_result$edges[edge_name, "au"]
      bp_val <- pv_result$edges[edge_name, "bp"]
    } else {
      au_val <- NA
      bp_val <- NA
    }
  } else {
    au_val <- NA
    bp_val <- NA
  }

  cluster_au$Cluster[k] <- k
  cluster_au$AU[k]      <- au_val
  cluster_au$BP[k]      <- bp_val
}

cat("\nCluster bootstrap support:\n")
print(cluster_au)

write.csv(cluster_au, output_csv, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_csv))
