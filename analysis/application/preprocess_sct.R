# Usage: Rscript analysis/application/preprocess_sct.R input.rda residuals.csv [object_name] [seed]
# The paper does not record the original Seurat/sctransform versions or explicit
# preprocessing seed. Preserve the original residual CSV for exact replay.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop("Provide input RDA and output residual CSV.")
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(future))
object_name <- if (length(args)>=3) args[[3]] else "sc"
run_seed <- if (length(args)>=4) as.integer(args[[4]]) else 123L
if (is.na(run_seed)) stop("Invalid seed")
if (file.exists(args[[2]])) stop("Output exists; use a new destination.")
env <- new.env()
load(args[[1]], envir=env)
if (!exists(object_name, envir=env, inherits=FALSE)) stop("Seurat object not found.")
sc <- get(object_name, envir=env)
genes <- c("ENSG00000197728", "ENSG00000087088")
get_layer <- function(object, assay, name) {
  if (utils::packageVersion("SeuratObject") >= "5.0.0") {
    layers <- SeuratObject::Layers(object, assay=assay, search=NA)
    if (!(name %in% layers)) {
      stop(paste("Expected one", name, "layer in assay", assay,
                 "of the prepared object. Combine split layers before running this script."))
    }
    SeuratObject::LayerData(object, assay=assay, layer=name)
  } else {
    GetAssayData(object, assay=assay, slot=name)
  }
}
counts <- get_layer(sc, "RNA", "counts")
keep <- Matrix::colSums(counts)>0
sc <- subset(sc, cells=colnames(counts)[keep])
plan(sequential)
set.seed(run_seed)
sc <- SCTransform(sc, assay="RNA", new.assay.name="SCT",
                  residual.features=genes, variable.features.n=2,
                  ncells=1000, conserve.memory=TRUE, verbose=FALSE,
                  seed.use=run_seed)
residual <- get_layer(sc, "SCT", "scale.data")[genes,,drop=FALSE]
dir.create(dirname(args[[2]]), recursive=TRUE, showWarnings=FALSE)
write.csv(as.data.frame(as.matrix(residual)), args[[2]], row.names=TRUE)
capture.output(list(seed=run_seed, sessionInfo()), file=paste0(args[[2]],".session.txt"))
# CD4 selection and genotype matching follow in analysis.application.prepare.
