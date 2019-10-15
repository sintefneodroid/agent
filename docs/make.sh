#!/usr/bin/env bash
for f in Makefile; do
  make -f "$f" html || exit
done
