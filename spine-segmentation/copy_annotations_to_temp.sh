mkdir /tmp/rhydian
cd /tmp/rhydian
echo "Copying MRI scans"
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/files-for-annotation/mri-niis /tmp/rhydian
echo "Copying MRI annotations"
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/files-for-annotation/mri-annotations /tmp/rhydian
