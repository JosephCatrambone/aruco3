
use std::collections::HashMap;  // TODO: Better HashMap impl.

use crate::dictionaries::ARDictionary;
use crate::dictionaries::AR_DICTIONARIES;
use crate::hamming_distance;

pub struct ARDetector {
	code_to_id: HashMap<u64, usize>,
	code_list: &'static [u64],
	tau: u8,
	mark_size: u8,
	num_bits: u8,
}

impl ARDetector {
	fn new_from_ar_dictionary(d: &ARDictionary) -> Self {
		let mut code_to_id = HashMap::new();
		for (i, code) in d.code_list.iter().enumerate() {
			code_to_id.insert(*code, i);
		}
		
		ARDetector {
			code_to_id: code_to_id,
			code_list: d.code_list,
			tau: if d.tau == 0 { 1 } else { d.tau }, // TODO: Calculate tau.
			mark_size: (d.num_bits as f64).sqrt().ceil() as u8 + 2,
			num_bits: d.num_bits,
		}
	}

	fn new_from_named_dict(code_name: &str) -> Result<Self, String> {
		if let Some(d) = AR_DICTIONARIES.get(code_name) {
			return Ok(Self::new_from_ar_dictionary(d));
		}
		panic!("TODO: code for this dict is not implemented.")
	}

	/// Search the dictionary for the nearest code (by hamming distance).
	/// Iterates across all elements in the code list, then returns the ID and the hamming distance
	fn find_nearest(&self, bits: u64) -> (usize, u8) {
		// The original implementation had this, which seems to do a search for an exact match, but we get that with the min hamming distance, too.
		/*
		  var val = '', i, j;
		  for (i = 0; i < bits.length; i++) {
		    var bitRow = bits[i];
		    for (j = 0; j < bitRow.length; j++) {
		      val += bitRow[j];
		    }
		  }
		  var minFound = this.codes[val];
		  if (minFound)
		    return {
		      id: minFound.id,
		      distance: 0
		    };
		*/
		// Build a default if we happen to not find anything.
		let mut min_id: usize = 0;
		let mut min_index: usize = 0;
		let mut min_code: u64 = 0;
		let mut min_distance: u8 = 0xFFu8;

		for (idx, c) in self.code_list.iter().enumerate() {
			let dist = hamming_distance(*c, bits);
			if dist < min_distance && dist < self.tau {
				min_distance = dist;
				min_index = idx;
				min_code = *c;
				min_id = *self.code_to_id.get(&c).expect("CRITICAL: code_list contained an entry that was not found in the code_to_id mapping.");
			}
		}
		
		(min_id, min_distance)
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_find_nearest() {
		let d = ARDetector::new_from_named_dict("ARUCO_DEFAULT");
		assert_eq!(hamming_distance(0x0 as u64, 0xFFFFFFFF_FFFFFFFF as u64), 64);
	}
}