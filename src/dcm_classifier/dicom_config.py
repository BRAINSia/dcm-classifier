# =========================================================================
#
#    Copyright NumFOCUS
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0.txt
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#  =========================================================================


"""
List of DICOM header columns that typically do not contain series-specific information and
can be safely dropped during data processing.

These columns are usually metadata or general patient/study information that do not provide
distinguishing details for identifying different scan types within a series. Removing them
can help reduce the dimensionality of the data and improve processing efficiency.

Attributes:
    drop_columns_with_no_series_specific_information (tuple): A tuple containing the names of
    DICOM header columns to be dropped from the dataset.

Example:
    Example usage to drop these columns from a DataFrame:
    df.drop(columns=drop_columns_with_no_series_specific_information, inplace=True)
"""
drop_columns_with_no_series_specific_information = (  # These are columns that have no ability to distinguish how to identify scan types
    "SpecificCharacterSet",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "ContentTime",
    "AccessionNumber",
    "Modality",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysician'sName",
    "StationName",
    "ProcedureCodeSequence",
    "StudyDescription",
    "InstitutionalDepartmentName",
    "PhysiciansofRecord",
    "PerformingPhysician'sName",
    "Operators'Name",
    "ReferencedStudySequence",
    "ReferencedSOPInstanceUID",
    "Patient'sName",
    "PatientID",
    "IssuerofPatientID",
    "Patient'sBirthDate",
    "Patient'sSex",
    "Patient'sAge",
    "Patient'sSize",
    "Patient'sWeight",
    "Patient'sAddress",
    "Patient'sTelephoneNumbers",
    "PatientComments",
    "BodyPartExamined",
    "TransmitCoilName",
    "StudyID",
    "FrameofReferenceUID",
    "FrameofReferenceUID",
    "PositionReferenceIndicator",
    # "SliceLocation",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "RequestedProcedureDescription",
    "AcquisitionTime",
    "RequestedProcedureCodeSequence",
    "StudyComments",
    "PerformedProcedureStepStartDate",
    "PerformedProcedureStepStartTime",
    "PerformedProcedureStepID",
    "PerformedProcedureStepDescription",
    "RequestAttributesSequence",
    "CommentsonthePerformedProcedureStep",
    "ModalitiesinStudy",
    "CodeValue",
    "CodingSchemeDesignator",
    "CodeMeaning",
    "LowR-RValue",
    "HighR-RValue",
    "IntervalsAcquired",
    "IntervalsRejected",
    "ReconstructionDiameter",
    "ReceiveCoilName",
    "AcquisitionDuration",
    "TemporalPositionIdentifier",
    "NumberofTemporalPositions",
    "NumberofStudyRelatedInstances",
    "LossyImageCompression",
    "PerformedProcedureStepEndDate",
    "PerformedProcedureStepEndTime",
    "PresentationLUTShape",
    "ReferencedImageSequence",
    "PatientIdentityRemoved",
    "InversionTime",
    "HeartRate",
    "CardiacNumberofImages",
    "TriggerWindow",
    "Laterality",
    "ImagesinAcquisition",
    "StackID",
    "In-StackPositionNumber",
    "PlanarConfiguration",
    "NumberofFrames",
    "LossyImageCompressionRatio",
    "TriggerTime",
)
