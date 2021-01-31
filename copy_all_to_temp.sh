mkdir /tmp/rhydian
cd /tmp/rhydian
echo "Copying dxa scans..."
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/dxas.tar /tmp/rhydian/dxas.tar
echo "Unzipping DXA scans"
tar -xf dxas.tar
echo "Copying MRI scans..."
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/SynthesisedMRISlices2.tar /tmp/rhydian/SynthesisedMRISlices2.tar
echo "Unzipping MRI scans"
tar -xf SynthesisedMRISlices2.tar 
echo "Copying DXA segmentations..."
rsync -ah --progress /scratch/shared/beegfs/rhydian/UKBiobank/dxa-segmentation3.tar /tmp/rhydian/dxa-segmentation3.tar
echo "Unzipping DXA segmentations"
tar -xf dxa-segmentation3.tar

