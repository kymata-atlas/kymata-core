#!/usr/local/bin/perl
use strict;
use warnings;

foreach my $participant ("0045", "0051", "0054", "0055", "0056", "0058", "0060", "0065", "0066", "0070", "0071", "0072", "0079", "0081","0082"){


	foreach my $m (1,2){

		open (my $infile, "<", "meg15_$participant/events_part$m\_raw.eve") or die $!;
		open (my $audiooutfile, ">", "meg15_$participant/events_part$m\_audio.eve") or die $!;
		open (my $visualoutfile, ">", "meg15_$participant/events_part$m\_visual.eve") or die $!;
	
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
				elsif ($condition == 3){
					#do audio	
					foreach my $k (1..400){	
						my $newsample = $sample + (($k-1)*1000) ;
						my $newtime = $time + ($k-1) ;
						print $audiooutfile "$newsample\t$newtime\t0\t$k\titem$k\n";	
					}	
				}

			}
		}

	}

	#close($infile); 
	#close($audiooutfile); 
	#close($visualoutfile); 

}
