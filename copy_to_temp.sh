mkdir -p $2
echo 'Copying MRIs...'
rsync -r --no-inc-recursive --info=progress2 $1'/mri-mid-corr-slices' $2
echo 'Copying DXAs...'
rsync -r --no-inc-recursive --info=progress2 $1'/dxas-processed' $2
