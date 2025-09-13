// Experiment Data Editor - JavaScript Logic

class ExperimentEditor {
    constructor() {
        this.currentDataset = {
            experiment_name: "Conformal Feedback Interpretation Experiment",
            description: "Test how model uncertainty changes with different feedback intensities and modalities",
            base_config: {
                // This will be populated when loading or creating scenarios
                input_context: {
                    current_action_description: "you are repositioning the user's arm during a therapy session.",
                    current_state: {
                        contact_forces: {
                            entire_arm: 2,
                            upper_arm: 1,
                            forearm: 1,
                            wrist: 2
                        },
                        joint_angles_deg: {
                            elbow: 135,
                            wrist: 135
                        }
                    },
                    current_comfort_threshold: {
                        current_comfort_threshold: {
                            entire_arm: { "2": 0.1, "3": 0.8, "4": 0.1 },
                            upper_arm: { "2": 0.1, "3": 0.8, "4": 0.1 },
                            forearm: { "2": 0.1, "3": 0.8, "4": 0.1 },
                            wrist: { "2": 0.1, "3": 0.8, "4": 0.1 }
                        },
                        current_comfortable_joint_range_deg: {
                            min: {
                                elbow: { "0": 0.6, "15": 0.3, "30": 0.1 },
                                wrist: { "0": 0.6, "15": 0.3, "30": 0.1 }
                            },
                            max: {
                                elbow: { "135": 0.1, "150": 0.3, "165": 0.6 },
                                wrist: { "135": 0.1, "150": 0.3, "165": 0.6 }
                            }
                        }
                    }
                }
            },
            scenarios: []
        };
        this.initializeAsync();
    }

    async initializeAsync() {
        this.initializeEventListeners();
        this.initializeDefaultValues();
    }

    initializeEventListeners() {
        // Config upload  
        document.getElementById('config-upload').addEventListener('change', this.onConfigUpload.bind(this));

        // Probability sliders
        this.initializeProbabilitySliders();

        // Action buttons (preview disabled to only update on save)
        document.getElementById('add-scenario-btn').addEventListener('click', this.addScenario.bind(this));
        document.getElementById('download-btn').addEventListener('click', this.downloadConfig.bind(this));

        // Normalize buttons
        document.querySelectorAll('.normalize-btn').forEach(btn => {
            btn.addEventListener('click', this.normalizeDistribution.bind(this));
        });

        // Auto-populate scenario name based on metadata
        ['verbal-intensity', 'facial-intensity', 'source-specificity'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', this.updateScenarioName.bind(this));
            }
        });
    }

    initializeProbabilitySliders() {
        // Body part sliders
        document.querySelectorAll('.prob-slider').forEach(slider => {
            slider.addEventListener('input', this.updateSliderValue.bind(this));
            slider.addEventListener('change', this.updateSumDisplay.bind(this));
        });

        // Angle sliders
        document.querySelectorAll('.angle-slider').forEach(slider => {
            slider.addEventListener('input', this.updateSliderValue.bind(this));
            slider.addEventListener('change', this.updateSumDisplay.bind(this));
        });
    }

    updateScenarioName() {
        const verbalIntensity = document.getElementById('verbal-intensity').value;
        const facialIntensity = document.getElementById('facial-intensity').value;
        const sourceSpecificity = document.getElementById('source-specificity').value;
        
        if (verbalIntensity && facialIntensity && sourceSpecificity) {
            const scenarioName = `${verbalIntensity}_${facialIntensity}_${sourceSpecificity}`;
            document.getElementById('scenario-name').value = scenarioName;
        }
    }

    async onExperimentSelect(event) {
        const experimentType = event.target.value;
        if (experimentType) {
            // Try to load the actual config file first, fallback to template
            const configLoaded = await this.tryLoadExperimentConfig(experimentType);
            if (!configLoaded) {
                this.loadExperimentTemplate(experimentType);
            }
            document.getElementById('scenario-form').style.display = 'block';
            this.updateIndependentVariableField(experimentType);
            await this.updateFeedbackInputType(experimentType);
        } else {
            document.getElementById('scenario-form').style.display = 'none';
            document.getElementById('independent-variable-group').style.display = 'none';
            document.getElementById('feedback-text-inputs').style.display = 'block';
            document.getElementById('feedback-dropdown-inputs').style.display = 'none';
        }
    }

    async tryLoadExperimentConfig(experimentType) {
        try {
            const configPath = `../configs/${experimentType}.json`;
            console.log('Attempting to load config from:', configPath);
            const response = await fetch(`${configPath}?t=${Date.now()}`);
            
            console.log('Fetch response status:', response.status, response.statusText);
            if (response.ok) {
                const config = await response.json();
                console.log('Successfully loaded config:', config);
                this.currentExperimentConfig = config;
                this.populateFormFromConfig();
                this.showStatus(`Loaded ${config.experiment_name} configuration`, 'success');
                return true;
            } else {
                console.log('Failed to load config, status:', response.status);
            }
        } catch (error) {
            console.log('Could not load config file, using template instead:', error.message);
        }
        console.log('Falling back to template');
        return false;
    }

    async onConfigUpload(event) {
        const file = event.target.files[0];
        if (file && file.type === 'application/json') {
            try {
                const text = await file.text();
                const loadedData = JSON.parse(text);
                
                // Handle new clean format
                if (loadedData.base_config && loadedData.scenarios) {
                    this.currentDataset = {
                        experiment_name: loadedData.experiment_name || "Loaded Experiment",
                        description: loadedData.description || "Loaded from file",
                        base_config: loadedData.base_config,
                        scenarios: loadedData.scenarios || []
                    };
                    
                    this.showStatus(`Loaded ${this.currentDataset.scenarios.length} scenarios successfully!`, 'success');
                    document.getElementById('scenario-form').style.display = 'block';
                    document.getElementById('download-btn').style.display = 'inline-block';
                    
                    // Show loaded scenarios
                    this.displayLoadedScenarios();
                    
                    // Update preview
                    const previewEl = document.getElementById('json-preview');
                    if (previewEl) {
                        previewEl.textContent = JSON.stringify(this.currentDataset, null, 2);
                    }
                    
                } else {
                    // Handle old format for backwards compatibility
                    this.currentExperimentConfig = loadedData;
                    this.showStatus('Config file loaded successfully!', 'success');
                    document.getElementById('scenario-form').style.display = 'block';
                    this.populateFormFromConfig();
                }
                
            } catch (error) {
                this.showStatus('Error loading config file: ' + error.message, 'error');
            }
        }
    }

    loadExperimentTemplate(experimentType) {
        console.log('Loading experiment template for:', experimentType);
        // Create base template based on experiment type
        this.currentExperimentConfig = {
            experiment_name: this.getExperimentName(experimentType),
            description: this.getExperimentDescription(experimentType),
            base_config: this.getBaseConfig(),
            scenarios: []
        };
        
        console.log('Created template config:', this.currentExperimentConfig);
        // Populate form with base config values from the template
        this.populateFormFromConfig();
    }

    getExperimentName(type) {
        const names = {
            'experiment_1_disagreement': 'Q1: Speech vs Facial Expression Disagreement',
            'experiment_2_ambiguity': 'Q2: Location Specificity Ambiguity', 
            'experiment_3_intensity': 'Q3: Adaptation to Discomfort Intensities',
            'experiment_4_modality': 'Q4: Facial Expression Modality Comparison',
            'experiment_5_uncertainty': 'Q5: Single-token vs Verbalized Uncertainty'
        };
        return names[type] || 'Custom Experiment';
    }

    getExperimentDescription(type) {
        const descriptions = {
            'experiment_1_disagreement': 'Test how model uncertainty changes when speech and facial expressions disagree',
            'experiment_2_ambiguity': 'Test model behavior with varying levels of location specificity',
            'experiment_3_intensity': 'Test how model adapts to different intensities of discomfort signals',
            'experiment_4_modality': 'Compare model responses to text vs image-based facial expressions',
            'experiment_5_uncertainty': 'Compare uncertainty measurements from token distributions vs model verbalized uncertainty'
        };
        return descriptions[type] || 'Custom experiment description';
    }

    updateIndependentVariableField(experimentType) {
        const variableGroup = document.getElementById('independent-variable-group');
        const variableLabel = document.querySelector('label[for="independent-variable"]');
        const variableHelp = document.getElementById('independent-variable-help');
        const variableSelect = document.getElementById('independent-variable');
        
        const variableInfo = {
            'experiment_1_disagreement': {
                label: 'Disagreement Type:',
                help: 'Combination of discomfort levels in verbal and facial feedback (no/mid/high discomfort)',
                options: [
                    { value: 'agreement_no_discomfort', text: 'Both No Discomfort (Agreement)' },
                    { value: 'agreement_mid_discomfort', text: 'Both Mid Discomfort (Agreement)' },
                    { value: 'agreement_high_discomfort', text: 'Both High Discomfort (Agreement)' },
                    { value: 'verbal_no_face_mid', text: 'Verbal No + Face Mid Discomfort' },
                    { value: 'verbal_no_face_high', text: 'Verbal No + Face High Discomfort' },
                    { value: 'verbal_mid_face_no', text: 'Verbal Mid + Face No Discomfort' },
                    { value: 'verbal_mid_face_high', text: 'Verbal Mid + Face High Discomfort' },
                    { value: 'verbal_high_face_no', text: 'Verbal High + Face No Discomfort' },
                    { value: 'verbal_high_face_mid', text: 'Verbal High + Face Mid Discomfort' }
                ]
            },
            'experiment_2_ambiguity': {
                label: 'Ambiguity Level:',
                help: 'Level of location specificity in feedback (low = specific, mid = moderate, high = vague)',
                options: [
                    { value: 'low', text: 'Low' },
                    { value: 'mid', text: 'Mid' },
                    { value: 'high', text: 'High' }
                ]
            },
            'experiment_3_intensity': {
                label: 'Intensity Level:',
                help: 'Level of discomfort intensity in feedback (low = mild, mid = moderate, high = severe)',
                options: [
                    { value: 'low', text: 'Low' },
                    { value: 'mid', text: 'Mid' },
                    { value: 'high', text: 'High' }
                ]
            },
            'experiment_4_modality': {
                label: 'Modality Level:',
                help: 'Input modality complexity (low = text description, high = image input)',
                options: [
                    { value: 'low', text: 'Low' },
                    { value: 'mid', text: 'Mid' },
                    { value: 'high', text: 'High' }
                ]
            },
            'experiment_5_uncertainty': {
                label: 'Uncertainty Level:',
                help: 'Expected uncertainty level in scenario (low = clear signals, mid = moderate, high = ambiguous/conflicting)',
                options: [
                    { value: 'low', text: 'Low' },
                    { value: 'mid', text: 'Mid' },
                    { value: 'high', text: 'High' }
                ]
            }
        };
        
        if (variableInfo[experimentType]) {
            variableGroup.style.display = 'block';
            variableLabel.textContent = variableInfo[experimentType].label;
            variableHelp.textContent = variableInfo[experimentType].help;
            
            // Update select options
            variableSelect.innerHTML = '<option value="">-- Select Type --</option>';
            variableInfo[experimentType].options.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option.value;
                optionElement.textContent = option.text;
                variableSelect.appendChild(optionElement);
            });
        } else {
            variableGroup.style.display = 'none';
        }
    }

    async updateFeedbackInputType(experimentType) {
        const textInputs = document.getElementById('feedback-text-inputs');
        const dropdownInputs = document.getElementById('feedback-dropdown-inputs');
        
        if (experimentType === 'experiment_1_disagreement') {
            // Show dropdowns for experiment 1
            textInputs.style.display = 'none';
            dropdownInputs.style.display = 'block';
            // Load facial expression images dynamically
            await this.loadFacialExpressionImages();
        } else {
            // Show text inputs for other experiments
            textInputs.style.display = 'block';
            dropdownInputs.style.display = 'none';
        }
    }

    async loadVerbalFeedbackOptions() {
        try {
            const response = await fetch('../assets/verbal.json');
            if (response.ok) {
                const verbalData = await response.json();
                const verySpecific = verbalData.veryspecific;
                
                this.verbalFeedbackOptions = {
                    'high': verySpecific.highdiscomfort || [],
                    'mid': verySpecific.middiscomfort || [],
                    'low': verySpecific.nodiscomfort || []
                };
                
                console.log('Loaded verbal feedback options:', this.verbalFeedbackOptions);
            } else {
                console.warn('Could not load verbal.json, using fallback options');
                this.createFallbackVerbalOptions();
            }
        } catch (error) {
            console.warn('Error loading verbal feedback options:', error);
            this.createFallbackVerbalOptions();
        }
    }

    createFallbackVerbalOptions() {
        this.verbalFeedbackOptions = {
            'high': [
                "ow that's too tight on my wrist",
                "you're grabbing my wrist way too tight",
                "my wrist feels like it's being crushed",
                "way too much pressure on my wrist",
                "you're digging hard into my wrist"
            ],
            'mid': [
                "my wrist hurts a little",
                "that's a bit tight on my wrist",
                "the pressure's just a bit too much on my wrist",
                "i'm feeling some pressure on my wrist",
                "the pressure's a bit uncomfortable on my wrist"
            ],
            'low': [
                "i'm fine",
                "yeah this is good",
                "everything's ok",
                "my wrist is good",
                "no pain at all"
            ]
        };
    }


    async loadFacialExpressionImages() {
        // Directly use the known image files from each directory
        this.facialExpressionImages = {
            'high': ['s001a.jpg', 's003a.jpg', 's011a.jpg', 's029a.jpg', 's032a.jpg'].map(filename => ({
                filename: filename,
                path: `assets/faceimgs/high/${filename}`
            })),
            'mid': ['s031a.jpg', 's041a.jpg', 's043a.jpg', 's046a.jpg', 's058a.jpg'].map(filename => ({
                filename: filename,
                path: `assets/faceimgs/mid/${filename}`
            })),
            'low': ['s001n.jpg', 's003n.jpg', 's026n.jpg', 's027n.jpg', 's044n.jpg'].map(filename => ({
                filename: filename,
                path: `assets/faceimgs/low/${filename}`
            }))
        };
        
        this.populateFacialExpressionDropdown();
    }


    populateFacialExpressionDropdown() {
        const dropdown = document.getElementById('facial-expression-dropdown');
        dropdown.innerHTML = '<option value="">-- Select Facial Expression Level --</option>';
        
        const categoryLabels = {
            'high': 'High Discomfort',
            'mid': 'Mid Discomfort', 
            'low': 'No/Low Discomfort'
        };
        
        // Add one option per category
        for (const [category, images] of Object.entries(this.facialExpressionImages)) {
            if (images.length > 0) {
                const option = document.createElement('option');
                option.value = category;
                option.textContent = categoryLabels[category];
                option.dataset.category = category;
                dropdown.appendChild(option);
            }
        }
        
        // No automatic label population - user sets thresholds manually
    }


    getBaseConfig() {
        // Use the base_config from the loaded experiment, or fallback to default
        if (this.currentExperimentConfig && this.currentExperimentConfig.base_config) {
            return this.currentExperimentConfig.base_config;
        }
        
        // Fallback default config if no experiment is loaded
        return {
            input_context: {
                current_action_description: "you are repositioning the user's arm during a therapy session.",
                current_state: {
                    contact_forces: {"entire_arm": 2, "upper_arm": 1, "forearm": 1, "wrist": 2},
                    joint_angles_deg: {"elbow": 165, "wrist": 165}
                },
                current_comfort_threshold: {
                    current_comfort_threshold: {
                        "entire_arm": {"2": 0.1, "3": 0.8, "4": 0.1},
                        "upper_arm": {"2": 0.1, "3": 0.8, "4": 0.1},
                        "forearm": {"2": 0.1, "3": 0.8, "4": 0.1},
                        "wrist": {"2": 0.1, "3": 0.8, "4": 0.1}
                    },
                    current_comfortable_joint_range_deg: {
                        min: {
                            "elbow": {"0": 0.6, "15": 0.3, "30": 0.1},
                            "wrist": {"0": 0.6, "15": 0.3, "30": 0.1}
                        },
                        max: {
                            "elbow": {"135": 0.1, "150": 0.3, "165": 0.6},
                            "wrist": {"135": 0.1, "150": 0.3, "165": 0.6}
                        }
                    }
                }
            }
        };
    }

    updateSliderValue(event) {
        const slider = event.target;
        const valueSpan = slider.nextElementSibling;
        valueSpan.textContent = parseFloat(slider.value).toFixed(2);
    }

    updateSumDisplay(event) {
        const slider = event.target;
        const section = slider.closest('.body-part-section, .joint-range-section');
        
        if (section) {
            const sliders = section.querySelectorAll('.prob-slider, .angle-slider');
            const sumDisplay = section.querySelector('.sum-value');
            
            let sum = 0;
            sliders.forEach(s => {
                sum += parseFloat(s.value);
            });
            
            sumDisplay.textContent = sum.toFixed(2);
            
            // Update sum display styling based on value
            const sumContainer = section.querySelector('.sum-display');
            sumContainer.className = 'sum-display';
            
            if (Math.abs(sum - 1.0) < 0.01) {
                sumContainer.classList.add('sum-perfect');
            } else if (sum > 1.1) {
                sumContainer.classList.add('sum-error');
            } else if (sum < 0.9 || sum > 1.05) {
                sumContainer.classList.add('sum-warning');
            }
        }
    }

    normalizeDistribution(event) {
        const button = event.target;
        const section = button.closest('.body-part-section, .joint-range-section');
        
        if (section) {
            const sliders = section.querySelectorAll('.prob-slider, .angle-slider');
            
            // Calculate current sum
            let sum = 0;
            sliders.forEach(slider => {
                sum += parseFloat(slider.value);
            });
            
            // Normalize if sum > 0
            if (sum > 0) {
                sliders.forEach(slider => {
                    const normalizedValue = parseFloat(slider.value) / sum;
                    slider.value = normalizedValue.toFixed(2);
                    this.updateSliderValue({target: slider});
                });
                this.updateSumDisplay({target: sliders[0]});
            } else {
                // If all zeros, set equal distribution
                const equalValue = (1.0 / sliders.length).toFixed(2);
                sliders.forEach(slider => {
                    slider.value = equalValue;
                    this.updateSliderValue({target: slider});
                });
                this.updateSumDisplay({target: sliders[0]});
            }
        }
    }

    // Helper function to remove zero values from distributions
    removeZeroValues(distribution) {
        const filtered = {};
        for (const [key, value] of Object.entries(distribution)) {
            if (value !== 0 && value !== 0.0) {
                filtered[key] = value;
            }
        }
        return filtered;
    }

    collectFormData() {
        const scenarioName = document.getElementById('scenario-name').value;
        const scenarioDescription = document.getElementById('scenario-description').value;
        
        // Collect experiment metadata
        const experimentMetadata = {
            verbal_intensity: document.getElementById('verbal-intensity').value,
            facial_intensity: document.getElementById('facial-intensity').value,
            source_specificity: document.getElementById('source-specificity').value
        };
        
        // Expert labels - body parts
        const bodyParts = ['entire_arm', 'upper_arm', 'forearm', 'wrist'];
        const labels = {};
        
        // Removed ask_clarification logic
        
        // Collect body part distributions (filter out zeros)
        bodyParts.forEach(part => {
            const section = document.querySelector(`[data-part="${part}"]`);
            const sliders = section.querySelectorAll('.prob-slider');
            const distribution = {};
            
            sliders.forEach(slider => {
                const level = slider.dataset.level;
                const value = parseFloat(slider.value);
                if (value !== 0 && value !== 0.0) {
                    distribution[level] = value;
                }
            });
            
            labels[part] = distribution;
        });
        
        // Collect joint range distributions in nested structure matching specification
        // Structure: { joint_range_min: { elbow: {...}, wrist: {...} }, joint_range_max: { elbow: {...}, wrist: {...} } }
        ['min', 'max'].forEach(rangeType => {
            const rangeData = {};
            
            ['elbow', 'wrist'].forEach(joint => {
                const section = document.querySelector(`[data-joint="${joint}"][data-range="${rangeType}"]`);
                if (section) {
                    const distribution = {};
                    
                    // Only include angles with non-zero values
                    for (let angle = 0; angle <= 180; angle += 15) {
                        const angleStr = angle.toString();
                        const slider = section.querySelector(`[data-angle="${angleStr}"]`);
                        if (slider) {
                            const value = parseFloat(slider.value);
                            if (value !== 0 && value !== 0.0) {
                                distribution[angleStr] = value;
                            }
                        }
                    }
                    
                    rangeData[joint] = distribution;
                }
            });
            
            // Store in nested format: joint_range_min, joint_range_max
            labels[`joint_range_${rangeType}`] = rangeData;
        });
        
        // Removed unused independent variable logic
        
        const scenarioData = {
            description: scenarioDescription || undefined,
            experiment_metadata: experimentMetadata,
            labels
        };
        
        // Remove unused independent variable logic
        
        return {
            scenarioName,
            scenarioData
        };
    }

    // previewJSON removed; preview now updates only after adding a scenario

    validateProbabilityDistributions(labels) {
        const bodyParts = ['entire_arm', 'upper_arm', 'forearm', 'wrist'];
        const jointRangeTypes = ['joint_range_min', 'joint_range_max'];
        const joints = ['elbow', 'wrist'];
        
        // Check body part distributions
        for (const part of bodyParts) {
            const distribution = labels[part];
            const sum = Object.values(distribution).reduce((a, b) => a + b, 0);
            
            if (Math.abs(sum - 1.0) > 0.01) {
                return {
                    valid: false,
                    message: `${part} probability distribution sums to ${sum.toFixed(2)}, should be 1.00`
                };
            }
        }
        
        // Check nested joint range distributions
        for (const rangeType of jointRangeTypes) {
            const rangeData = labels[rangeType];
            if (rangeData) {
                for (const joint of joints) {
                    const distribution = rangeData[joint];
                    if (distribution && Object.keys(distribution).length > 0) {
                        const sum = Object.values(distribution).reduce((a, b) => a + b, 0);
                        
                        if (Math.abs(sum - 1.0) > 0.01) {
                            return {
                                valid: false,
                                message: `${rangeType}.${joint} probability distribution sums to ${sum.toFixed(2)}, should be 1.00`
                            };
                        }
                        
                        // Validate that all present angles are valid (0-180° in 15° increments)
                        for (const angleStr of Object.keys(distribution)) {
                            const angle = parseInt(angleStr);
                            if (isNaN(angle) || angle < 0 || angle > 180 || angle % 15 !== 0) {
                                return {
                                    valid: false,
                                    message: `${rangeType}.${joint} contains invalid angle: ${angleStr}°`
                                };
                            }
                        }
                    }
                }
            }
        }
        
        return { valid: true };
    }

    generateAllScenarios(formData) {
        // Extract categories from the form data
        const verbalFeedbackDescription = formData.scenarioData.received_feedback.verbal_feedback.description;
        const facialExpressionDescription = formData.scenarioData.received_feedback.facial_expression.description;
        
        // Parse the categories (format: "category_category")
        const verbalCategory = verbalFeedbackDescription.replace('_category', '');
        const facialCategory = facialExpressionDescription.replace('_category', '');
        
        if (!verbalCategory || !facialCategory) {
            throw new Error('No categories selected for experiment 1');
        }
        
        // Get all options for the selected categories
        const verbalOptions = this.verbalFeedbackOptions[verbalCategory] || [];
        const facialOptions = this.facialExpressionImages[facialCategory] || [];
        
        if (verbalOptions.length === 0 || facialOptions.length === 0) {
            throw new Error('Selected categories have no available options');
        }
        
        // Generate all combinations
        const scenarios = [];
        let scenarioIndex = 1;
        
        for (const verbalOption of verbalOptions) {
            for (const facialOption of facialOptions) {
                const scenarioData = {
                    ...formData.scenarioData,
                    received_feedback: {
                        verbal_feedback: { description: verbalOption },
                        facial_expression: { 
                            modality: 'image', 
                            description: facialOption.path 
                        }
                    }
                };
                
                const scenarioName = `${formData.scenarioName}_${scenarioIndex}`;
                scenarios.push({
                    name: scenarioName,
                    ...scenarioData
                });
                
                scenarioIndex++;
            }
        }
        
        return scenarios;
    }

    addScenario() {
        try {
            const formData = this.collectFormData();
            
            if (!formData.scenarioName.trim()) {
                this.showStatus('Please enter a scenario name.', 'error');
                return;
            }
            
            // Validate probability distributions
            const validationResult = this.validateProbabilityDistributions(formData.scenarioData.labels);
            if (!validationResult.valid) {
                this.showStatus('Validation error: ' + validationResult.message, 'error');
                return;
            }
            
            // Add scenario to unified dataset
            const scenarioWithName = { 
                name: formData.scenarioName, 
                ...formData.scenarioData 
            };
            this.currentDataset.scenarios.push(scenarioWithName);
            
            // Show download button
            document.getElementById('download-btn').style.display = 'inline-block';
            
            this.showStatus(`Scenario "${formData.scenarioName}" added successfully!`, 'success');
            
            // Update JSON preview
            const previewEl = document.getElementById('json-preview');
            if (previewEl) {
                previewEl.textContent = JSON.stringify(this.currentDataset, null, 2);
            }
            
            // Clear only text inputs for next scenario; keep sliders unchanged
            this.clearTextInputs();
            
        } catch (error) {
            this.showStatus('Error adding scenario: ' + error.message, 'error');
        }
    }

    downloadConfig() {
        if (!this.currentDataset || this.currentDataset.scenarios.length === 0) {
            this.showStatus('No scenarios to download.', 'error');
            return;
        }
        
        const datasetJSON = JSON.stringify(this.currentDataset, null, 2);
        const blob = new Blob([datasetJSON], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const filename = 'unified_scenario_dataset.json';
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showStatus(`Configuration downloaded as ${filename}`, 'success');
    }

    getIndependentVariableFieldName(experimentType) {
        const fieldNames = {
            'experiment_1_disagreement': 'disagreement_type',
            'experiment_2_ambiguity': 'ambiguity_level',
            'experiment_3_intensity': 'intensity_level',
            'experiment_4_modality': 'modality_level',
            'experiment_5_uncertainty': 'uncertainty_level'
        };
        return fieldNames[experimentType] || null;
    }

    // Clear only text inputs and checkbox; preserve sliders
    clearTextInputs() {
        document.getElementById('scenario-name').value = '';
        document.getElementById('scenario-description').value = '';
    }

    clearForm() {
        // Clear text inputs
        document.getElementById('scenario-name').value = '';
        document.getElementById('scenario-description').value = '';
        document.getElementById('verbal-feedback').value = '';
        document.getElementById('facial-description').value = '';
        document.getElementById('independent-variable').value = '';
        
        // Reset checkbox
        
        // Reset all sliders to 0
        document.querySelectorAll('.prob-slider, .angle-slider').forEach(slider => {
            slider.value = 0;
            this.updateSliderValue({target: slider});
        });
        
        // Update all sum displays
        document.querySelectorAll('.body-part-section, .joint-range-section').forEach(section => {
            const firstSlider = section.querySelector('.prob-slider, .angle-slider');
            if (firstSlider) {
                this.updateSumDisplay({target: firstSlider});
            }
        });
        
        // Hide preview
        document.getElementById('preview-section').style.display = 'none';
    }

    clearFormOnly() {
        // Clear text inputs only (don't hide preview)
        document.getElementById('scenario-name').value = '';
        document.getElementById('scenario-description').value = '';
        document.getElementById('verbal-feedback').value = '';
        document.getElementById('facial-description').value = '';
        document.getElementById('independent-variable').value = '';
        
        // Reset checkbox
        
        // Reset all sliders to 0 (they will be populated by config)
        document.querySelectorAll('.prob-slider, .angle-slider').forEach(slider => {
            slider.value = 0;
            this.updateSliderValue({target: slider});
        });
        
        // Update all sum displays
        document.querySelectorAll('.body-part-section, .joint-range-section').forEach(section => {
            const firstSlider = section.querySelector('.prob-slider, .angle-slider');
            if (firstSlider) {
                this.updateSumDisplay({target: firstSlider});
            }
        });
    }

    populateFormFromConfig() {
        if (!this.currentExperimentConfig || !this.currentExperimentConfig.base_config) {
            this.showStatus('Config loaded - ready to add new scenarios', 'info');
            return;
        }

        // Clear form first
        this.clearFormOnly();

        const baseConfig = this.currentExperimentConfig.base_config;
        
        // Extract comfort threshold values from base config
        const comfortThresholds = baseConfig.input_context?.current_comfort_threshold?.current_comfort_threshold;
        if (comfortThresholds) {
            this.populateComfortThresholds(comfortThresholds);
        }
        
        // Extract joint range values from base config  
        const jointRanges = baseConfig.input_context?.current_comfort_threshold?.current_comfortable_joint_range_deg;
        if (jointRanges) {
            this.populateJointRanges(jointRanges);
        }
        
        this.showStatus('Form initialized with base configuration values', 'info');
    }

    initializeDefaultValues() {
        // Wait a bit for DOM to be fully ready
        setTimeout(() => {
            if (this.currentDataset.base_config) {
                const baseConfig = this.currentDataset.base_config;
                
                // Initialize comfort thresholds from base config
                const comfortThresholds = baseConfig.input_context?.current_comfort_threshold?.current_comfort_threshold;
                if (comfortThresholds) {
                    this.populateComfortThresholds(comfortThresholds);
                }
                
                // Initialize joint ranges from base config  
                const jointRanges = baseConfig.input_context?.current_comfort_threshold?.current_comfortable_joint_range_deg;
                if (jointRanges) {
                    this.populateJointRanges(jointRanges);
                }
                
                console.log('Initialized form with default base config values');
                
                // Show the scenario form since we have base config
                document.getElementById('scenario-form').style.display = 'block';
            }
        }, 500); // Give time for DOM to load
    }

    displayLoadedScenarios() {
        const infoDiv = document.getElementById('loaded-scenarios-info');
        const listDiv = document.getElementById('loaded-scenarios-list');
        
        if (this.currentDataset.scenarios.length > 0) {
            listDiv.innerHTML = this.currentDataset.scenarios.map(scenario => {
                const metadata = scenario.experiment_metadata;
                return `<div class="loaded-scenario">
                    <strong>${scenario.name}</strong> - 
                    Verbal: ${metadata.verbal_intensity}, 
                    Facial: ${metadata.facial_intensity}, 
                    Specificity: ${metadata.source_specificity}
                </div>`;
            }).join('');
            infoDiv.style.display = 'block';
        } else {
            infoDiv.style.display = 'none';
        }
    }

    populateComfortThresholds(comfortThresholds) {
        const bodyParts = ['entire_arm', 'upper_arm', 'forearm', 'wrist'];
        
        bodyParts.forEach(part => {
            const section = document.querySelector(`[data-part="${part}"]`);
            if (section && comfortThresholds[part]) {
                const distribution = comfortThresholds[part];
                console.log(`Populating ${part} with:`, distribution);
                
                // Update each slider for levels 1-5
                for (let level = 1; level <= 5; level++) {
                    const slider = section.querySelector(`[data-level="${level}"]`);
                    if (slider && distribution[level.toString()] !== undefined) {
                        slider.value = distribution[level.toString()];
                        this.updateSliderValue({target: slider});
                    }
                }
                
                // Update sum display
                const firstSlider = section.querySelector('.prob-slider');
                if (firstSlider) {
                    this.updateSumDisplay({target: firstSlider});
                }
            } else {
                console.log(`No data found for body part: ${part}`);
            }
        });
    }

    populateJointRanges(jointRanges) {
        // Read from schema: { min: { elbow|wrist }, max: { elbow|wrist } }
        const minElbow = jointRanges?.min?.elbow || null;
        const minWrist = jointRanges?.min?.wrist || null;
        const maxElbow = jointRanges?.max?.elbow || null;
        const maxWrist = jointRanges?.max?.wrist || null;

        if (!minElbow && !minWrist && !maxElbow && !maxWrist) {
            console.log('No joint range data found');
            return;
        }

        console.log('Populating joint ranges with:', { min: { elbow: minElbow, wrist: minWrist }, max: { elbow: maxElbow, wrist: maxWrist } });

        // Generate all angles from 0 to 180 in 15° increments
        const allAngles = [];
        for (let angle = 0; angle <= 180; angle += 15) {
            allAngles.push(angle.toString());
        }

        // Populate each joint section separately
        const jointData = {
            elbow: { min: minElbow, max: maxElbow },
            wrist: { min: minWrist, max: maxWrist }
        };
        
        ['elbow', 'wrist'].forEach(joint => {
            ['min', 'max'].forEach(rangeType => {
                const section = document.querySelector(`[data-joint="${joint}"][data-range="${rangeType}"]`);
                const data = jointData[joint][rangeType];
                
                if (section && data) {
                    allAngles.forEach(angle => {
                        const slider = section.querySelector(`[data-angle="${angle}"]`);
                        if (slider && data[angle] !== undefined) {
                            slider.value = data[angle];
                            this.updateSliderValue({ target: slider });
                        }
                    });
                    
                    // Update sum display
                    const firstSlider = section.querySelector('.angle-slider');
                    if (firstSlider) {
                        this.updateSumDisplay({ target: firstSlider });
                    }
                }
            });
        });
    }

    showStatus(message, type = 'info') {
        const statusContainer = document.getElementById('status-messages');
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message status-${type}`;
        statusDiv.textContent = message;
        
        statusContainer.appendChild(statusDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.parentNode.removeChild(statusDiv);
            }
        }, 5000);
    }
}

// Utility functions for probability distribution helpers
class ProbabilityDistributionHelpers {
    static createUniformDistribution(keys) {
        const value = 1.0 / keys.length;
        const distribution = {};
        keys.forEach(key => {
            distribution[key] = value;
        });
        return distribution;
    }
    
    static createBiasedDistribution(keys, peakKey, concentration = 2.0) {
        const distribution = {};
        const peakIndex = keys.indexOf(peakKey);
        
        keys.forEach((key, index) => {
            const distance = Math.abs(index - peakIndex);
            distribution[key] = Math.exp(-distance / concentration);
        });
        
        return this.normalize(distribution);
    }
    
    static normalize(distribution) {
        const sum = Object.values(distribution).reduce((a, b) => a + b, 0);
        if (sum === 0) return distribution;
        
        const normalized = {};
        Object.keys(distribution).forEach(key => {
            normalized[key] = distribution[key] / sum;
        });
        return normalized;
    }
    
    static validateDistribution(distribution, tolerance = 0.01) {
        const sum = Object.values(distribution).reduce((a, b) => a + b, 0);
        return Math.abs(sum - 1.0) <= tolerance;
    }
}

// Initialize the editor when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.experimentEditor = new ExperimentEditor();
    window.probHelpers = ProbabilityDistributionHelpers;
    
    // Update scenario name after DOM is fully loaded
    setTimeout(() => {
        if (window.experimentEditor && window.experimentEditor.updateScenarioName) {
            window.experimentEditor.updateScenarioName();
        }
    }, 100);
});

// Add quick-fill buttons for common distributions
document.addEventListener('DOMContentLoaded', () => {
    // Add quick-fill buttons to each distribution section
    setTimeout(() => {
        document.querySelectorAll('.body-part-section, .joint-range-section').forEach(section => {
            const buttonsDiv = document.createElement('div');
            buttonsDiv.className = 'quick-fill-buttons';
            buttonsDiv.style.marginTop = '10px';
            buttonsDiv.style.display = 'flex';
            buttonsDiv.style.gap = '5px';
            buttonsDiv.style.flexWrap = 'wrap';
            
            // Uniform distribution button
            const uniformBtn = document.createElement('button');
            uniformBtn.type = 'button';
            uniformBtn.className = 'btn-quick-fill';
            uniformBtn.textContent = 'Uniform';
            uniformBtn.style.padding = '4px 8px';
            uniformBtn.style.fontSize = '0.8rem';
            uniformBtn.style.background = '#17a2b8';
            uniformBtn.style.color = 'white';
            uniformBtn.style.border = 'none';
            uniformBtn.style.borderRadius = '3px';
            uniformBtn.style.cursor = 'pointer';
            
            uniformBtn.addEventListener('click', () => {
                const sliders = section.querySelectorAll('.prob-slider, .angle-slider');
                const equalValue = (1.0 / sliders.length).toFixed(2);
                sliders.forEach(slider => {
                    slider.value = equalValue;
                    window.experimentEditor.updateSliderValue({target: slider});
                });
                window.experimentEditor.updateSumDisplay({target: sliders[0]});
            });
            
            // Clear button
            const clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.className = 'btn-quick-fill';
            clearBtn.textContent = 'Clear';
            clearBtn.style.padding = '4px 8px';
            clearBtn.style.fontSize = '0.8rem';
            clearBtn.style.background = '#6c757d';
            clearBtn.style.color = 'white';
            clearBtn.style.border = 'none';
            clearBtn.style.borderRadius = '3px';
            clearBtn.style.cursor = 'pointer';
            
            clearBtn.addEventListener('click', () => {
                const sliders = section.querySelectorAll('.prob-slider, .angle-slider');
                sliders.forEach(slider => {
                    slider.value = 0;
                    window.experimentEditor.updateSliderValue({target: slider});
                });
                window.experimentEditor.updateSumDisplay({target: sliders[0]});
            });
            
            buttonsDiv.appendChild(uniformBtn);
            buttonsDiv.appendChild(clearBtn);
            
            // Insert before normalize button
            const normalizeBtn = section.querySelector('.normalize-btn');
            normalizeBtn.parentNode.insertBefore(buttonsDiv, normalizeBtn);
        });
    }, 100);

});
