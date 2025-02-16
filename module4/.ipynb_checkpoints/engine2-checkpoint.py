class LowShotCounting(nn.Module):
    def __init__(
        self,
        num_iterations=3,
        embed_dim=256,
        temperature=0.1,
        backbone_type='swin',
        num_exemplars=3
    ):
        super().__init__()
        print("\nInitializing LowShotCounting...")
        print(f"Parameters: iterations={num_iterations}, embed_dim={embed_dim}, temperature={temperature}")
        print(f"Backbone: {backbone_type}, num_exemplars: {num_exemplars}")
        
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim
        self.num_exemplars = num_exemplars
        
        # Initialize components with memory tracking
        print("\nInitializing model components...")
        try:
            self.encoder = DensityEncoder(min_dim=embed_dim)
            print("✓ Encoder initialized")
            
            self.feature_enhancer = FeatureEnhancer(dims=(256, 512, 1024))
            print("✓ Feature Enhancer initialized")
            
            self.exemplar_learner = ExemplarFeatureLearning(
                embed_dim=embed_dim,
                num_iterations=num_iterations
            )
            print("✓ Exemplar Learner initialized")
            
            self.matcher = ExemplarImageMatching(
                embed_dim=embed_dim,
                temperature=temperature
            )
            print("✓ Matcher initialized")
            
            self.decoders = nn.ModuleList([
                DensityRegressionDecoder(input_channels=num_exemplars) 
                for _ in range(num_iterations)
            ])
            print("✓ Decoders initialized")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def extract_features(self, x):
        """Extract features from input image"""
        print(f"\nExtracting features:")
        print(f"Input shape: {x.shape}")
        try:
            features = self.encoder(x)
            print(f"Extracted feature shape: {features.shape}")
            print(f"Feature stats - Min: {features.min():.3f}, Max: {features.max():.3f}, Mean: {features.mean():.3f}")
            return features
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise

    def extract_exemplar_patches(self, image, bboxes):
        print(f"\nExtracting exemplar patches:")
        print(f"Image shape: {image.shape}")
        print(f"Bboxes shape: {bboxes.shape}")
        
        B, C, H, W = image.shape
        K = bboxes.shape[1]
        patch_size = 128

        try:
            # Debug bboxes values
            print("\nBbox statistics:")
            print(f"Min coords: {bboxes.min(dim=1)[0]}")
            print(f"Max coords: {bboxes.max(dim=1)[0]}")
            
            bboxes = bboxes.round().long()
            patches = []
            
            for b in range(B):
                batch_patches = []
                for k in range(K):
                    x1, y1, x2, y2 = bboxes[b, k]
                    print(f"\nBatch {b}, Exemplar {k}:")
                    print(f"Original coords: ({x1}, {y1}, {x2}, {y2})")
                    
                    # Boundary checks
                    x1 = torch.clamp(x1, 0, W-1)
                    x2 = torch.clamp(x2, x1+1, W)
                    y1 = torch.clamp(y1, 0, H-1)
                    y2 = torch.clamp(y2, y1+1, H)
                    print(f"Clamped coords: ({x1}, {y1}, {x2}, {y2})")
                    
                    patch = image[b:b+1, :, y1:y2, x1:x2]
                    print(f"Patch shape before resize: {patch.shape}")
                    
                    patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    print(f"Patch shape after resize: {patch.shape}")
                    
                    batch_patches.append(patch)
                patches.append(torch.cat(batch_patches, dim=0))
            
            final_patches = torch.stack(patches)
            print(f"\nFinal patches shape: {final_patches.shape}")
            return final_patches
            
        except Exception as e:
            print(f"Error in exemplar patch extraction: {str(e)}")
            print(f"Error location - Batch: {b}, Exemplar: {k}")
            raise

    def process_exemplars(self, bboxes, image):
        """Process exemplar patches"""
        print("\nProcessing exemplars:")
        print(f"Input image shape: {image.shape}")
        print(f"Input bboxes shape: {bboxes.shape}")
        
        try:
            exemplar_patches = self.extract_exemplar_patches(image, bboxes)
            print(f"Extracted patches shape: {exemplar_patches.shape}")
            
            B, K, C, H, W = exemplar_patches.shape
            exemplars_flat = exemplar_patches.view(B * K, C, H, W)
            print(f"Flattened exemplars shape: {exemplars_flat.shape}")
            
            exemplar_features = self.encoder(exemplars_flat)
            print(f"Encoded exemplars shape: {exemplar_features.shape}")
            
            exemplar_features = F.adaptive_avg_pool2d(exemplar_features, (1, 1))
            exemplar_features = exemplar_features.view(B, K, -1)
            print(f"Final exemplar features shape: {exemplar_features.shape}")
            
            # Memory check
            print("\nCurrent GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
            
            return exemplar_features
            
        except Exception as e:
            print(f"Error in exemplar processing: {str(e)}")
            raise

    def forward(self, image, bboxes):
        print("\n" + "="*50)
        print("Starting forward pass...")
        print(f"Input image shape: {image.shape}")
        print(f"Input bboxes shape: {bboxes.shape}")
        
        try:
            # Extract features
            image_features = self.extract_features(image)
            B, C, H, W = image_features.shape
            print(f"Extracted image features shape: {image_features.shape}")
            
            # Process exemplars
            exemplar_features = self.process_exemplars(bboxes, image)
            print(f"Processed exemplar features shape: {exemplar_features.shape}")
            
            current_exemplars = exemplar_features
            all_density_maps = []
            
            # Iterative prediction
            for i in range(self.num_iterations):
                print(f"\nIteration {i+1}/{self.num_iterations}")
                
                try:
                    # Prepare image features
                    img_features_flat = image_features.permute(0, 2, 3, 1)
                    img_features_flat = img_features_flat.reshape(B, H * W, C)
                    print(f"Flattened image features shape: {img_features_flat.shape}")
                    
                    # Update exemplars
                    current_exemplars = self.exemplar_learner(
                        img_features_flat,
                        current_exemplars,
                        bboxes
                    )
                    print(f"Updated exemplars shape: {current_exemplars.shape}")
                    
                    # Generate similarity maps
                    similarity_maps = self.matcher(
                        image_features,
                        current_exemplars
                    )
                    print(f"Similarity maps shape: {similarity_maps.shape}")
                    print(f"Similarity stats - Min: {similarity_maps.min():.3f}, Max: {similarity_maps.max():.3f}")
                    
                    # Predict density map
                    density_map = self.decoders[i](similarity_maps)
                    print(f"Density map shape: {density_map.shape}")
                    print(f"Density stats - Min: {density_map.min():.3f}, Max: {density_map.max():.3f}, Sum: {density_map.sum():.3f}")
                    
                    # Memory check after each iteration
                    print("\nGPU Memory after iteration:")
                    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
                    
                    if density_map.shape[2:] != image_features.shape[2:]:
                        density_map = F.interpolate(
                            density_map,
                            size=image_features.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    all_density_maps.append(density_map)
                    
                except Exception as e:
                    print(f"Error in iteration {i+1}: {str(e)}")
                    raise
            
            print("\nForward pass completed successfully")
            return all_density_maps
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise