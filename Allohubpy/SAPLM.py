import esm
import torch
import torch.nn.functional as F
import pandas as pd


class SAPLM:

    def __init__(self, fragment_size):
        """
        Initializes the handler for Protein Language Models (ESM2_650M), loads the model and sets the fragment size

        Args:
            fragment_size (int): Number of aminoacids that constitute one fragment.

        """

        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.eval()
        self.sequence = ""
        self.probabilities = []
        self.fragment_size = fragment_size

    def set_sequence(self, sequence):
        """
        Sets the sequence to use and computes probabilities for each position using the language model.

        Args:
            sequence (string): aminoacid sequence to process. 

        Raises:
            ValueError: If the sequence is not a string
    
        Example:
            SAPLM.set_sequence("IQLYPGKNAHHCACTQINKK")

        """
        
        if not isinstance(sequence, str):
            raise ValueError("Sequence must be a string of aminoacids")
        
        self.sequence = sequence
        probs = self._run_inference()
        self.probabilities = probs


    def _run_inference(self):
        """
        Return the probabilities for the loaded sequence

        Returns:
            probabilities (tensor): batch_size=0, sequence_length (minus start and end token), possible tokens. 
            The probabilities for non amino acid tokens are set to 0.
        """

        batch_converter = self.alphabet.get_batch_converter()

        batch_labels, batch_strs, batch_tokens = batch_converter([("protein1", self.sequence)])

        amino_acid_positions = [self.alphabet.get_idx(c) for c in "ACDEFGHIKLMNPQRSTVWY"] 

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens)

        # Get logits    
        logits = results["logits"]
        
        # Initialize a tensor to store probabilities
        probs = torch.zeros_like(logits)  

        # Get only logits for aminoacids
        for i in range(logits.shape[1]): 
            position_logits = logits[0, i, :]  
            amino_acid_logits = position_logits[amino_acid_positions]  

            # Softmax over the amino acid logits
            amino_acid_probs = F.softmax(amino_acid_logits, dim=0)

            # Store results
            probs[0, i, amino_acid_positions] = amino_acid_probs


        # return only the positions for aminoacids (tokens )
        return probs[0, 1:-1, :].numpy()
    

    def fragment_likelihoods(self, fragment, offset=0):
        """
        Extracts the amino acids and associated likelihoods for a given fragment id.

        Args:
            fragment (int): fragment index to process. 
            offset (int): Displacement between fragments and the full length protein (usually due to unmodelled residues).
    
        Returns:
            results_df (pandas DataFrame): with header []"Amino acid position", "Amino acid", "Likelihood"].

        """

        results = {"Amino acid position":[], "Amino acid":[], "Likelihood": []}

        if self.sequence:
            for i in range(fragment, fragment + self.fragment_size):
                results["Amino acid position"].append(i+1+offset)
                aa = self.sequence[i]
                results["Amino acid"].append(aa)
                results["Likelihood"].append(self.probabilities[i,self.alphabet.get_idx(aa)])

        else:
            print("No sequence found, please use .set_sequence()")

        results_df = pd.DataFrame(results)

        return results_df