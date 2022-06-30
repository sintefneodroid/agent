@ECHO OFF

pushd %~dp0

git push --all
git push --all pything
git push --tags pything

popd
