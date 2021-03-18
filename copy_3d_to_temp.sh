mkdir /tmp/rhydian
cd /tmp/rhydian
echo "Copying MRI scans"
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/Cleaned3DMRIsHalfResolution.tar /tmp/rhydian/Cleaned3DMRIsHalfResolution.tar
echo "Unzipping MRI scans"
tar -xf Cleaned3DMRIsHalfResolution.tar
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/TransferredMRISegmentations.tar /tmp/rhydian/TransferredMRISegmentations.tar
echo "Unzipping DXA segmentations"
tar -xf TransferredMRISegmentations.tar 
