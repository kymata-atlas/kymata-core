#!/usr/local/bin/perl
use strict;
use warnings;

foreach my $participant ("0173", "0178", "0193", "0195", "0213", "0219", "0226", "0230", "0239", "0436"){


	foreach my $m (1,2){

		open (my $infile, "<", "meg14_$participant/events_part$m\_pythontest.eve") or die $!;
		open (my $audiooutfile, ">", "meg14_$participant/events_part$m\_audio\_pythontest.eve") or die $!;
	
		print $infile "\n";
			
		my $i=0;
		while (<$infile>){
			my $line = "$_";
			(my $sample, my $time, my $mask, my $condition) = split(/\t/, $line);
			if ($i == 0){
				print $visualoutfile "$sample\t$time\t$mask\t$i\tpsuedoitem\n";
				print $audiooutfile "$sample\t$time\t$mask\t$i\tpsuedoitem\n";
				$i++;
			}
			else{
				if ($condition == 2){
					#do visual
					print $visualoutfile "$sample\t$time\t$mask\t$i\titem$i\n";
					if ($i == 400){
						$i = 0;
					}
					$i++;	
				}

			}
		}

	}

	#close($infile); 
	#close($audiooutfile); 
	#close($visualoutfile); 

}
