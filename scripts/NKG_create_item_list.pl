#!/usr/local/bin/perl
use strict;
use warnings;

my $numberOfItems = $ARGV[0];

open MYFILE, ">items.txt" or die $!;

for (my $i=1; $i <= $numberOfItems; $i++) {
	print MYFILE "item$i\n";
}

close(MYFILE); 

