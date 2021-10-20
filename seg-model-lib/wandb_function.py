import wandb

def wandbInit(project, config, run_name):
    wandb.init(
        project=project, 
        reinit=True,
        config=config
    )
    wandb.run.name = run_name
    wandb.run.save()
    
def wandbWrite(epoch, loss, mIoU, val_loss, val_mIoU, IoU_by_class, summary=False):
    wandb.log({
        "train": 
            {"loss": loss, "mIoU": mIoU}, 
        "val": 
            {"loss": val_loss, "mIoU": val_mIoU},
        "IoU by Class": IoU_by_class
    })
    # wandb.run.summary["IoU by class"] = IoU_by_class
    if summary:
        wandb.run.summary.update({
            "epoch": epoch,
            "val.loss": val_loss,
            "val.mIoU": val_mIoU,
            "IoUClass": IoU_by_class
        })